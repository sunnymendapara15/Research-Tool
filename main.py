import os
import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import html

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "news-research")

# Streamlit page setup
st.title("ChatBot: News Research Tool")
st.sidebar.title("News Article URLs")

# Initialize session state
if "urls" not in st.session_state:
    st.session_state.urls = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Input for URLs
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url and url not in st.session_state.urls:
        st.session_state.urls.append(url)

# Process URLs button
process_url_clicked = st.sidebar.button("Process URLs")

# Placeholder for processing status
main_placeholder = st.empty()

# Initialize the LLM
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), 
               model_name="Llama3-70b-8192",
               temperature=0.1)

# Process URLs when button is clicked
if process_url_clicked and st.session_state.urls:
    try:
        # Load data
        main_placeholder.text("Data Loading...Started...✅✅✅")
        loader = UnstructuredURLLoader(urls=st.session_state.urls)
        data = loader.load()
        time.sleep(1)  # Brief pause for visibility
        
        # Log documents loaded per URL
        url_doc_counts = {}
        for doc in data:
            url = doc.metadata.get('source', 'Unknown')
            url_doc_counts[url] = url_doc_counts.get(url, 0) + 1
        main_placeholder.text(f"Loaded {len(data)} documents from {len(st.session_state.urls)} URLs")
        for url, count in url_doc_counts.items():
            main_placeholder.text(f"- {url}: {count} document(s)")
        time.sleep(1)  # Brief pause for visibility
        
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=300,  # Reduced for finer granularity
            chunk_overlap=150  # Increased overlap for better context
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        main_placeholder.text(f"Created {len(docs)} document chunks")
        time.sleep(1)  # Brief pause for visibility
        
        # Create embeddings and save to Pinecone index
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        # Clear existing Pinecone index to avoid duplicates
        if pinecone_index_name in [index.name for index in pc.list_indexes()]:
            pc.delete_index(pinecone_index_name)
        pc.create_index(name=pinecone_index_name, dimension=384, metric='cosine', spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}})
        time.sleep(5)  # Wait for index creation
        vector_store = PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
        st.session_state.vector_store = vector_store
        time.sleep(2)  # Match original delay
        
        main_placeholder.text("Pinecone index created and populated")
        main_placeholder.text("Ready to chat! Ask your question below.")
    except Exception as e:
        main_placeholder.error(f"Error processing URLs: {str(e)}")

# Chat interface
st.header("Chat with the News Research Tool")

# Chat history container
chat_container = st.container()

# Input form for Enter key submission
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input("Question:", key=f"query_{st.session_state.input_key}")
    submit_button = st.form_submit_button("Send")

# Process query when submitted (via button or Enter key)
if submit_button and query:
    if st.session_state.vector_store:
        try:
            # Create the retrieval chain with a retriever that fetches more documents
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, 
                retriever=st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 12}  # Increased to fetch more documents
                )
            )
            
            # Run the query
            result = chain({"question": query}, return_only_outputs=True)
            # Log retrieved documents for debugging
            retrieved_docs = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 12}
            ).get_relevant_documents(query)
            main_placeholder.text(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
            for doc in retrieved_docs:
                main_placeholder.text(f"- Retrieved from: {doc.metadata.get('source', 'Unknown')}")
                main_placeholder.text(f"  Content snippet: {doc.page_content[:100]}...")
            
            # Check if answer is "I don't know" and provide fallback
            answer = result["answer"].strip()
            if "don't know" in answer.lower():
                answer += "\n\nNote: The system retrieved documents, but the answer may not fully address the query. Try rephrasing the question or checking if the URL content matches the query."
            
            # Append to chat history
            st.session_state.chat_history.append({
                "question": query,
                "answer": answer,
                "sources": result.get("sources", "")
            })
            
            # Increment input key to reset the input field
            st.session_state.input_key += 1
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Display chat history
with chat_container:
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(
            """
            <div style="padding: 10px; border-radius: 10px; margin: 5px;">
                <strong>You:</strong> {}
            </div>
            """.format(html.escape(chat['question'])),
            unsafe_allow_html=True
        )
        # Bot message
        st.markdown(
            """
            <div style="padding: 10px; border-radius: 10px; margin: 5px;">
                <strong>Bot:</strong> {}
            </div>
            """.format(html.escape(chat['answer'])),
            unsafe_allow_html=True
        )
        # Sources, if available
        if chat.get("sources"):
            sources_html = ""
            for source in chat['sources'].split('\n'):
                if source.strip() and source.startswith(('http://', 'https://')):
                    sources_html += f'<a href="{source}" style="color: #1E90FF; text-decoration: underline;">{source}</a><br>'
            if sources_html:
                st.markdown(
                    """
                    <div style="padding: 10px; border-radius: 10px; margin: 5px;">
                        <strong>Sources:</strong><br>{}
                    </div>
                    """.format(sources_html),
                    unsafe_allow_html=True
                )
    # Auto-scroll to the bottom
    st.markdown(
        """
        <script>
            var element = document.querySelector('[data-testid="stVerticalBlock"] > div:last-child');
            element.scrollTop = element.scrollHeight;
        </script>
        """,
        unsafe_allow_html=True
    )