import streamlit as st

from web_search_rag import app


def run_interface(app):
    logo_url = "src/logo.png"
    st.image(logo_url, width=50)
    st.title("OpenCV Knowledge Assistant")
    query = st.text_input("", key="query_input", placeholder="Enter your question")

    if query:
        with st.spinner("Analyzing..."):
            initial_state = {"question": query}
            response = app.invoke(initial_state)
            
            # Main response
            st.markdown(response['generation'].response)
            
            # Collapsible context section
            with st.expander("Retrieval Context ▼", expanded=False):
                if 'documents' in response:
                    st.write("**📄 Relevant Documents:**")
                    for i, doc in enumerate(response['documents'], 1):
                        st.markdown(f"""
                        <div style="
                            padding: 10px; 
                            background: #000000;
                            border-radius: 5px; 
                            margin: 5px 0;
                            font-size: 0.9em;
                            color: #ffffff;
                        ">
                        <b>Document {i}:</b><br>
                        {doc.text}
                        </div>
                        """, unsafe_allow_html=True)
                        st.divider()
                
                if 'web_results' in response:
                    st.write("**🌐 Web Results:**")
                    for i, page in enumerate(response['web_results'], 1):
                        st.markdown(f"""
                        <div style="
                            padding: 10px;
                            background: #000000;
                            border-radius: 5px;
                            margin: 5px 0;
                            font-size: 0.9em;
                            color: #ffffff;
                        ">
                        <b>Result {i}: {page.node.metadata.get('title', 'No Title')}</b><br>
                        {page.node.metadata.get('content', 'No content')}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write("🔗 Source:", page.node.metadata.get('url', 'No URL available'))
                        st.divider()


if __name__ == "__main__":
    run_interface(app)
