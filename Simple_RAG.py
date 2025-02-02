import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler

# Configuration Constants
PDF_SOURCE_PATH = "./source"
EMBEDDING_MODEL = "deepseek-r1:1.5b"  # Better for embeddings
LLM_MODEL = "deepseek-r1:1.5b"
CHUNK_SIZE = 1024  # Align with model context window
CHUNK_OVERLAP = 256

# Custom streaming callback handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(
            f'<div class="answer-box">{self.text}</div>', 
            unsafe_allow_html=True
        )

# Custom CSS for styling
custom_css = """
<style>
    .main {background-color: #f9f9f9; padding: 2rem; border-radius: 0.5rem;}
    .sidebar .sidebar-content {background-image: linear-gradient(#2e7bcf, #2e7bcf); color: white;}
    h1 {color: #2e7bcf;}
    .expander-header {font-size: 1.1rem; font-weight: bold;}
    .answer-box {
        background: #2d2d2d;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Cached system initialization for better performance"""
    try:
        loader = DirectoryLoader(
            PDF_SOURCE_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True  # Preserve document position
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    prompt_template = """Use the following context to answer the question. 
If you don't know the answer, say you don't know. 
Keep the answer concise and technical.

Context: {context}
Question: {question}
Helpful Answer:"""
    
    QA_PROMPT = ChatPromptTemplate.from_template(prompt_template)

    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.3,  # Control creativity
        num_ctx=CHUNK_SIZE,  # Match context window
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.4
        }
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",  
        output_key="result"  
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        input_key="query",  
        output_key="result"  
    )
    
    return qa_chain

# -------------------------
# Sidebar for navigation and instructions
st.sidebar.title("About This App")
st.sidebar.info(
    """
    This app lets you ask technical questions about your PDFs.
    
    **How it works:**
    - PDFs from the source folder are loaded and processed.
    - Your question is used to retrieve relevant chunks.
    - An LLM generates a concise technical answer.
    
    Enjoy exploring your documents!
    """
)

# Main layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("Enhanced PDF Reader")
st.write("Ask technical questions about your documents. Type your query below and hit Submit.")

# Use a form to capture input
with st.form(key="qa_form"):
    user_query = st.text_input("Enter your question:")
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    qa_chain = initialize_system()
    with st.spinner("Analyzing documents..."):
        try:
            # Set up columns before processing
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Answer:")
                answer_box = st.empty()
            
            # Initialize stream handler
            stream_handler = StreamHandler(answer_box)
            
            # Process query with streaming
            result = qa_chain.invoke(
                {"query": user_query},
                {"callbacks": [stream_handler]}
            )

            # Show sources in expander
            with col2:
                with st.expander("Source References", expanded=True):
                    for doc in result["source_documents"]:
                        st.write(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
                        
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
            
st.markdown('</div>', unsafe_allow_html=True)