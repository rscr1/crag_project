import gc
import os
import json
from typing import Any, List, TypedDict
from typing_extensions import NotRequired

import tiktoken
import torch

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, END
from llama_index.core import get_response_synthesizer, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document, NodeWithScore, TextNode, BaseNode
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from transformers import AutoModel


class MyEmbeddings(BaseEmbedding):
    def __init__(
        self, 
        model_name='jinaai/jina-embeddings-v3',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def _get_query_embedding(self, query: str) -> List[float]:
        self._model = self._model.to('cuda:7')
        embedding = self._model.encode(query, task="retrieval.query").tolist()
        clear_cache()
        return embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        self._model = self._model.to('cuda:6')
        embedding = self._model.encode(text, task="retrieval.passage").tolist()
        clear_cache()
        return embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        self._model = self._model.to('cuda:7')
        embeddings = [self._model.encode(t, task="retrieval.passage").tolist() for t in texts]
        clear_cache()
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        self._model = self._model.to('cuda:6')
        embedding = self._get_query_embedding(query)
        clear_cache()
        return embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        self._model = self._model.to('cuda:7')
        embedding = self._get_text_embedding(text)
        clear_cache()
        return embedding


class GraphState(TypedDict):
    """Represents the state of our RAG workflow"""
    question: str
    documents: NotRequired[List]
    generation: NotRequired[str]
    web_results: NotRequired[List]



class VectorStoreManager:
    def __init__(self, doc_path, chunk_size, chunk_overlap):
        embed_model = MyEmbeddings()
        nodes = self.parse_plat_doc(doc_path)
        self.index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    def parse_plat_doc(self, doc_path: str) -> list[BaseNode]:
        with open(doc_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        basic_options = data["basic_options"]
        chunks = []

        current_id = 1
        for option in basic_options:
            options_list = basic_options[option]
            for option_dict in options_list:
                text = json.dumps(option_dict, ensure_ascii=False)
                node = TextNode(text=text, id_=str(current_id))
                current_id += 1
                chunks.append(node)
        return chunks

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()

def web_search(query: str, num_results=3) -> List[Document]:
    """Perform web search and return results as Documents."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    
    documents = []
    for r in results:
        link = r.get('href', 'No URL available')
        title = r.get('title', 'No Title')
        content = r.get('body', 'No Content')
        text = f"Title: {title}\nContent: {content}"
        documents.append(Document(text=text, metadata={"url": link, "title": title, "content": content}))

    return documents


def retrieve_from_vector_store(state: GraphState) -> GraphState:
    """Retrieve documents from vector store"""
    print("---RETRIEVING FROM VECTOR STORE---")
    nodes = retriever.retrieve(state["question"])
    return {"documents": nodes}


def grade_documents(state: GraphState) -> GraphState:
    """Grade retrieved documents for relevance using the reranker."""
    print("---GRADING DOCUMENTS---")
    documents = state["documents"]
    
    reranked_nodes = reranker.postprocess_nodes(
        nodes=documents,
        query_str=state["question"]
    )

    filtered_docs = [
        doc for doc in reranked_nodes 
        if doc.score is None or doc.score >= similarity_cutoff
    ]
    
    return {"documents": filtered_docs}


def web_search_retrieval(state: GraphState) -> GraphState:
    """Perform web search when needed"""
    print("---PERFORMING WEB SEARCH---")
    web_docs = web_search(state["question"])
    web_nodes = []

    for doc in web_docs:
        node = NodeWithScore(
            node=doc,
            score=1.0,
            embedding=None
        )
        web_nodes.append(node)
    
    return {"web_results": web_nodes}


def generate_response(state: GraphState) -> GraphState:
    """Generate final response using available context"""
    print("---GENERATING RESPONSE---")
    
    documents = state.get("documents", [])
    web_results = state.get("web_results", [])
    all_context = documents + web_results
    
    if not all_context:
        return {"generation": "I don't have enough context to answer this question."}
    
    response = response_synthesizer.synthesize(
        nodes=all_context,
        query=state["question"]
    )
    return {"generation": response}


def should_web_search(state: GraphState) -> str:
    """Decide whether to use web search"""
    # if not state.get('documents', []):
    #     return "web_search"
    return "generate"


load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

input_path = "platform_docs/plat_doc.json"
model = "gpt-4o-mini"
choice_batch_size=15
top_n = 5
chunk_size=256
chunk_overlap=64
similarity_top_k=15
similarity_cutoff=0.3


# llm = OpenAI(model=model, api_key=api_key, api_base='https://api.proxyapi.ru/openai/v1')
llm = Ollama(model="rscr/vikhr_nemo_12b:latest", request_timeout=60.0, base_url=base_url, format="str", temperature=0.5)
index = VectorStoreManager(input_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap).index


RANKGPT_RERANK_PROMPT_TMPL = (
    "Search Query: {query}. \nRank the {num} passages above "
    "based on their relevance to the search query. The passages "
    "should be listed in descending order using identifiers. "
    "The most relevant passages should be listed first. "
    "The output format should be [] > [], e.g., [1] > [2]. "
    "Only response the ranking results, "
    "do not say any word or explain."
)
RANKGPT_RERANK_PROMPT = PromptTemplate(
    RANKGPT_RERANK_PROMPT_TMPL, prompt_type=PromptType.RANKGPT_RERANK
)
reranker = RankGPTRerank(top_n=top_n, llm=llm)
response_synthesizer = get_response_synthesizer(llm=llm)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=similarity_top_k,
    vector_store_query_mode=VectorStoreQueryMode.DEFAULT
    )

# Create the workflow graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve_from_vector_store)
workflow.add_node("grade", grade_documents)
workflow.add_node("web_search", web_search_retrieval)
workflow.add_node("generate", generate_response)

# Add edges
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    should_web_search,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Set entry point
workflow.set_entry_point("retrieve")

# Compile the workflow
app = workflow.compile()