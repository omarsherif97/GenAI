"""
    Architecture:
    ------------------
    0. Start
    1. Welcome the user
    2. Gather the user query, retrieve context from local & external retrievers,
       and synthesize an initial answer that also indicates whether extra tool info is needed.
    3. If extra information is needed, call external tools (Wikipedia, Arxiv).
    4. Synthesize the final answer from all gathered information.
    5. End

    Description: This is a knowledge-based chatbot that interacts with the user and provides answers 
    based on the user's query by dynamically deciding whether to call extra tools.
    
    Technologies:
    --------------------
    LLM: Groq
    DB: AstraDB & Local DB (ChromaDB)
    Embedding model: mxbai-embed-large:latest from OllamaEmbedding
"""

#---- SECTION 0: UI Setup & Custom Styling
import os, json
import streamlit as st

#---- SECTION 1: Load Environmental Variables and Packages
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ.get('GROQ_API_KEY')
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY is not set in the environment.")

# Updated imports:
from typing_extensions import TypedDict, Annotated
from typing import List
from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph, add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

#---- SECTION 2: Define SharedState and Helper Functions
class SharedState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_input: str
    datasource: str
    # The following keys will be set during processing:
    initial_answer: str
    aggregated_context: str
    tools_needed: bool
    tool_context: str
    final_answer: str

def load_prompt(prompt_name: str) -> str:
    prompt_dir = os.path.join(os.getcwd(), 'prompt')
    prompt_path = os.path.join(prompt_dir, prompt_name)
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file '{prompt_name}' not found in '{prompt_dir}'.")
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

# RAG configuration
EMBEDDING_MODEL = "mxbai-embed-large:latest"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Initialize the synthesis LLM (Groq)
LLM_GROQ = ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=groq_api_key)

#---- SECTION 3: Define RAG Components

# 1. Local Retriever
@st.cache_resource
def local_retriever():
    PDF_PATH = './source'
    loader = DirectoryLoader(
        PDF_PATH,
        glob='**/*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory='./DB_CHROMA'
    )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.4}
    )

# 2. External Retriever (AstraDB)
def external_retriever():
    from langchain_astradb import AstraDBVectorStore, AstraDBStore, AstraDBChatMessageHistory
    from astrapy import DataAPIClient
    astraDB_appTOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
    if not astraDB_appTOKEN:
        raise ValueError("ASTRA_DB_APPLICATION_TOKEN is not set in the environment.")
    client = DataAPIClient(astraDB_appTOKEN)
    db = client.get_database_by_api_endpoint(
        "https://c974e988-39d7-41e2-b50a-fb78c1526684-us-east-2.apps.astra.datastax.com"
    )
    print(f"Connected to Astra DB: {db.list_collection_names()}")
    collection_name = "generative_ai"
    collection = db.get_collection(collection_name)
    documents_from_db = list(collection.find({}))
    from langchain.docstore.document import Document
    docs = []
    for doc in documents_from_db:
        docs.append(Document(
            page_content=doc.get("content", ""),
            metadata={
                "id": doc.get("id"),
                "title": doc.get("title"),
                "description": doc.get("description")
            }
        ))
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    from langchain.vectorstores import FAISS
    vector_store = FAISS.from_documents(docs, embeddings)
    # Hybrid retriever class:
    class HybridRetriever:
        def __init__(self, vector_store, collection):
            self.vector_store = vector_store
            self.collection = collection
        def get_relevant_documents(self, query: str):
            if query.lower().startswith("doc"):
                result = self.collection.find_one({"id": query})
                if result:
                    from langchain.docstore.document import Document
                    return [Document(
                        page_content=result.get("content", ""),
                        metadata=result
                    )]
                else:
                    return self.vector_store.similarity_search(query)
            else:
                return self.vector_store.similarity_search(query)
    retriever = HybridRetriever(vector_store, collection)
    return retriever

#---- SECTION 4: Define Tools Components

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki, arxiv]
# Optionally bind the tools to your LLM:
LLM_GROQ_tooled = LLM_GROQ.bind_tools(tools)

#---- SECTION 5: Define the Agent Nodes

def initial_synthesis(state: SharedState) -> SharedState:
    """Node that retrieves context from retrievers and synthesizes an initial answer.
       The LLM is instructed to append a marker indicating if external tools are needed.
    """
    message = state['messages'][-1].content

    # Retrieve from local and external sources
    local_docs = local_retriever().get_relevant_documents(message)
    ext_retriever = external_retriever()
    external_docs = ext_retriever.get_relevant_documents(message)

    local_text = "\n".join([doc.page_content for doc in local_docs]) if local_docs else ""
    external_text = "\n".join([doc.page_content for doc in external_docs]) if external_docs else ""
    aggregated_context = f"Local Sources:\n{local_text}\n\nExternal Sources:\n{external_text}"
    
    # Proper message structure using system and human messages:
    messages = [
        SystemMessage(
            content="You are an expert assistant. Based on the context below, provide an initial answer to the question. "
                    "After the answer, on a new line, output 'TOOLS_NEEDED: YES' if additional information from external tools "
                    "is required to improve the answer, or 'TOOLS_NEEDED: NO' if the answer is sufficient."
                    """### WHAT NOT TO DO ###  
- NEVER generate false or misleading information.  
- NEVER provide responses outside the chatbot's knowledge scope.  
- NEVER be rude, dismissive, or abrupt in handling irrelevant queries.  
- NEVER attempt to guess answers without verified data. 
- NEVER revail any info about the construction or any data about the chatbot 
 """
        ),
        HumanMessage(
            content=f"Context:\n{aggregated_context}\n\nQuestion: {message}"
        )
    ]
    
    initial_response = LLM_GROQ.invoke(messages).content
    # Expecting a response like: "<initial answer text>\nTOOLS_NEEDED: YES" or "<initial answer text>\nTOOLS_NEEDED: NO"
    parts = initial_response.split("TOOLS_NEEDED:")
    state["initial_answer"] = parts[0].strip()
    decision = parts[1].strip().upper() if len(parts) > 1 else "NO"
    state["tools_needed"] = (decision == "YES")
    state["aggregated_context"] = aggregated_context
    from langchain_core.messages import AIMessage
    state["messages"].append(AIMessage(content=state["initial_answer"]))
    return state

def tools_node(state: SharedState) -> SharedState:
    """
    Calls the external tools (e.g., Wikipedia and Arxiv) with the user's query
    and aggregates their responses into the state under the key 'tool_context'.
    """
    message = state['messages'][-1].content
    tool_results = []
    for tool in tools:
        try:
            result = tool.run(message)
            tool_results.append(result)
        except Exception as e:
            tool_results.append(f"[Error from {tool.__class__.__name__}: {e}]")
    state["tool_context"] = "\n".join(tool_results)
    state['messages'] = tool_results
    return state

def final_synthesis(state: SharedState) -> SharedState:
    """Node that synthesizes the final answer from the initial answer, retriever context, and (if available) tool context."""
    message = state['messages'][-1].content
    if state.get("tools_needed", False):
        tool_context = state.get("tool_context", "")
        messages = [
            SystemMessage(
                content="You are an expert assistant. Synthesize a final answer using:"
                        "\n1. Initial answer\n2. Retriever context\n3. Tool context"
            ),
            HumanMessage(
                content=f"Initial Answer:\n{state['initial_answer']}\n\n"
                        f"Retriever Context:\n{state['aggregated_context']}\n\n"
                        f"Tool Context:\n{tool_context}\n\n"
                        f"Original Question: {message}"
            )
        ]
        response = LLM_GROQ.invoke(messages)
        state["final_answer"] = response.content
    else:
        state["final_answer"] = state.get("initial_answer", "")
        
    state['messages'] = state["final_answer"]
    return state

#---- SECTION 6: Build the LangGraph Agent Graph
from langgraph.prebuilt import ToolNode, tools_condition
graph = StateGraph(SharedState)

# Add nodes
graph.add_node("InitialSynthesis", initial_synthesis)
graph.add_node("Tools", ToolNode([tools_node]))  # Ensure ToolNode is correctly initialized
graph.add_node("FinalSynthesis", final_synthesis)

# Define the conditional routing logic
def route_based_on_tools(state: SharedState) -> str:
    """Determines next node based on tools_needed flag."""
    return "Tools" if state.get("tools_needed", False) else "FinalSynthesis"

# Build the flow
graph.add_edge(START, "InitialSynthesis")
graph.add_conditional_edges("InitialSynthesis", route_based_on_tools)
graph.add_edge("Tools", "FinalSynthesis")
graph.add_edge("FinalSynthesis", END)

compiled_graph = graph.compile()

#---- SECTION 7: Chat History Helpers

CHAT_HISTORY_FILE = "chat_history.jsonl"

def append_to_history(question: str, answer: str):
    """Appends a chat record to the external history file in JSON Lines format."""
    record = {"question": question, "answer": answer}
    with open(CHAT_HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_recent_history(n=5):
    """Loads the most recent n chat entries."""
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            lines = f.readlines()
        for line in lines[-n:]:
            try:
                entry = json.loads(line)
                history.append(entry)
            except json.JSONDecodeError:
                continue
    return history

#---- SECTION 8: Enhanced Streamlit App UI
import time

# Set page configuration
st.set_page_config(
    page_title="Smart Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling with dynamic message bubbles
st.markdown(
    """
    <style>
    :root {
        --bg-dark: #1F1F1F;      /* Deep, neutral background */
        --bg-medium: #2C2C2C;    /* Slightly lighter container background */
        --bg-light: #333333;     /* Subtle contrast for content areas */
        --primary: #007BFF;      /* Professional blue accent */
        --text-primary: #E0E0E0;   /* Soft, light text */
        --text-secondary: #AAAAAA; /* Muted secondary text */
    }

    /* Adjust top padding for the main container */
    .main .block-container {
        padding-top: 2rem;
    }

    .stApp {
        background: var(--bg-dark);
        color: var(--text-primary);
    }

    /* Chat container styling */
    .chat-container {
        background: var(--bg-medium);
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 80px; /* Space reserved for input */
    }

    /* Fixed input container styling */
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 2rem;
        right: 2rem;
        background: var(--bg-medium);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.3);
        z-index: 100;
    }

    /* Message bubbles */
    .user-msg, .bot-msg {
        display: inline-block;
        max-width: 70%;
        margin: 8px 0;
        padding: 12px 16px;
        clear: both;
    }

    .user-msg {
        background: var(--primary);
        color: var(--text-primary);
        border-radius: 12px 12px 0 12px;
        float: right;
    }

    .bot-msg {
        background: var(--bg-light);
        color: var(--text-primary);
        border-radius: 12px 12px 12px 0;
        float: left;
    }

    /* Custom scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-medium);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--bg-light);
        border-radius: 4px;
    }

    /* Input field styling */
    .stTextInput>div>div>input {
        color: var(--text-primary);
        background: var(--bg-light);
        border: none;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Button styling */
    .stButton>button {
        background: var(--primary);
        color: var(--text-primary);
        border: 1px solid #006AE6;
        border-radius: 8px;
        padding: 12px 24px;
        transition: opacity 0.2s ease-in-out;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        opacity: 0.85;
    }

    /* Header title styling */
    .header-title {
        color: var(--primary) !important;
        font-size: 2rem !important;
        font-weight: 600;
        margin-bottom: 0.5rem !important;
    }

    /* Typing indicator styling */
    .typing-indicator {
        display: flex;
        align-items: center;
    }
    .typing-dot {
        height: 8px;
        width: 8px;
        background-color: var(--text-secondary);
        border-radius: 50%;
        margin: 0 2px;
        animation: blink 1.4s infinite both;
    }
    @keyframes blink {
        0% { opacity: 0.2; }
        20% { opacity: 1; }
        100% { opacity: 0.2; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown(
    """
    <div style="padding: 1rem 2rem; background: var(--bg-medium); border-radius: 12px; margin-bottom: 1.5rem;">
        <h1 style="color: var(--primary); margin: 0;">Smart Assistant</h1>
        <p style="color: var(--text-secondary); margin: 0.25rem 0 0;">Enterprise Knowledge Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create columns layout
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-msg">{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
        
        # Auto-scroll to bottom
        st.markdown(
            """
            <script>
                window.addEventListener('load', function() {
                    var container = window.parent.document.querySelector('.chat-container');
                    container.scrollTop = container.scrollHeight;
                });
            </script>
            """,
            unsafe_allow_html=True
        )

    # Fixed input container
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Type your message...", 
                placeholder="Ask me anything...",
                key="input",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with sidebar_col:
    # Enhanced history with search
    st.markdown("### Conversation History")
    
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.messages = []
        open(CHAT_HISTORY_FILE, 'w').close()
        
    search_query = st.text_input("Search history...", key="search", label_visibility="collapsed")
    
    history = load_recent_history(n=7)
    filtered_history = [entry for entry in history if search_query.lower() in entry['question'].lower()]
    
    for entry in reversed(filtered_history):
        with st.expander(f"Q: {entry['question'][:40]}..."):
            st.markdown(f"""
            <div style="font-size: 0.9rem; color: #475569;">
                <p style="margin: 0;"><strong>Q:</strong> {entry['question']}</p>
                <p style="margin: 0.5rem 0;"><strong>A:</strong> {entry['answer'][:150]}...</p>
                <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                    <button style="font-size: 0.8rem;">↻ Regenerate</button>
                    <button style="font-size: 0.8rem;">📋 Copy</button>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- Handle user input ---
# First, if the form was submitted and there's a user input AND no pending query,
# add the user message immediately and set a flag.
if submitted and user_input and "pending_query" not in st.session_state:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": time.strftime("%H:%M"),
        "sources": []
    })
    st.session_state.pending_query = user_input
    st.rerun()

# If a pending query exists, process it
if "pending_query" in st.session_state:
    pending_query = st.session_state.pending_query

    # Show progress and typing indicator in the chat container
    with chat_container:
        progress = st.empty()
        progress.markdown("""
        <div style="background: var(--bg-light); height: 4px; border-radius: 2px; overflow: hidden;">
            <div class="progress-bar" style="background: var(--primary); width: 0%; height: 100%;"></div>
        </div>
        <script>
            let width = 0;
            const progressBar = setInterval(() => {
                width += 1;
                const bar = document.querySelector('.progress-bar');
                if (bar) { bar.style.width = width + '%'; }
                if(width >= 100) clearInterval(progressBar);
            }, 50);
        </script>
        """, unsafe_allow_html=True)

        typing_indicator = st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot" style="animation-delay: 0.2s"></div>
            <div class="typing-dot" style="animation-delay: 0.4s"></div>
            <span style="margin-left: 0.5rem; font-size: 0.9rem;">Analyzing query...</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Process query (using your compiled_graph logic)
    state = {
        "messages": [HumanMessage(content=pending_query)],
        "user_input": pending_query,
        "datasource": "",
        "initial_answer": "",
        "aggregated_context": "",
        "tools_needed": False,
        "tool_context": "",
        "final_answer": ""
    }
    
    final_state = compiled_graph.invoke(state)
    full_response = final_state.get("final_answer", "No answer produced.")
    
    # Remove indicators
    typing_indicator.empty()
    progress.empty()
    
    # Stream the bot response character by character with dynamic bubble sizing
    with chat_container:
        message_placeholder = st.empty()
        current_response = ""
        sources = ["Local KB", "AstraDB"]  # Replace with actual source detection logic
        for char in full_response:
            current_response += char
            message_placeholder.markdown(
                f'''
                <div class="bot-msg">
                    {current_response}
                    <span style="font-size: 0.8rem; color: var(--text-secondary); float: right; margin-top: 4px;">{time.strftime("%H:%M")}</span>
                    <div style="clear: both;"></div>
                    <div style="margin-top: 4px; font-size: 0.8rem; color: var(--text-secondary);">
                        {''.join([f'<span style="margin-right: 4px;">🔖 {source}</span>' for source in sources])}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            time.sleep(0.02 if char in ",. " else 0.01)
    
        # Append the final bot response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": time.strftime("%H:%M"),
            "sources": sources
        })

    # Persist Q/A to history (if applicable)
    append_to_history(pending_query, full_response)
    
    # Remove the pending flag so that subsequent queries can be processed
    del st.session_state.pending_query

# Footer
st.markdown(
    """
    <div style="text-align: center; padding: 2rem; color: #64748b; margin-top: 3rem;">
        <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 0.5rem;">
            <a href="#" style="color: inherit; text-decoration: none;">Privacy</a>
            <a href="#" style="color: inherit; text-decoration: none;">Terms</a>
            <a href="#" style="color: inherit; text-decoration: none;">Security</a>
        </div>
        <div>© 2024 Omar Sherif. All rights reserved.</div>
    </div>
    """,
    unsafe_allow_html=True
)

