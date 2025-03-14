# Knowledge Assistant

## Overview
The Knowledge Assistant is a web application designed to provide answers to user queries by leveraging a Retrieval-Augmented Generation (RAG) approach. This application utilizes various libraries and models to perform web searches, retrieve relevant documents, and generate responses based on the context provided.

## Example Outputs
Here are some examples of the application in action:

**In-domain (without web-search)**:
   ![alt text](/src/example1.png)
   ![alt text](/src/example2.png)

**Out-of-fomain (with web-search)**:
    ![alt text](/src/example3.png)
    ![alt text](/src/example4.png)

## Libraries Used
The following libraries are utilized in this project:

- **Streamlit**: For creating the web interface.
- **Torch**: For deep learning model operations.
- **Transformers**: For loading and using pre-trained models.
- **Ollama**: for inference LLM.
- **DuckDuckGo Search**: For performing web searches.
- **LangGraph**: For managing the workflow of the RAG process.
- **LlamaIndex**: For handling document embeddings and retrieval.
- **Tiktoken**: For tokenization of text inputs.
- **SQLite**: For storing chat history in a local database.

## Key Entities in RAG
The application implements several key entities in the RAG framework:

- **MyEmbeddings**: A custom embedding class that generates embeddings for queries and documents using a pre-trained model.
- **VectorStoreManager**: Manages the vector store, loading data, and creating embeddings for documents.
- **GraphState**: Represents the state of the RAG workflow, including the user's question, retrieved documents, and generated responses.
- **Workflow Graph**: A directed graph that defines the sequence of operations (nodes) and their connections (edges) in the RAG process.

## CRAG Method Implementation
The application implements the CRAG (Conditional Retrieval-Augmented Generation) method, which enhances the response generation process by conditionally deciding whether to perform a web search or generate a response based on the retrieved documents' relevance. The workflow consists of the following steps:

1. **Retrieve**: Fetch relevant documents from the vector store based on the user's query.
2. **Grade**: Evaluate the relevance of the retrieved documents using a reranker.
3. **Web Search**: If necessary, perform a web search to gather additional context.
4. **Generate**: Synthesize a final response using the available context from both retrieved documents and web search results.


