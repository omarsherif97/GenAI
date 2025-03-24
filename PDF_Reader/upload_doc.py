import os
import asyncio
from typing import Dict, List, Any, TypedDict, Optional
from dotenv import load_dotenv
import base64
import requests
import json
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
from io import BytesIO

# Load environment variables
load_dotenv()

# -- Core Document Processing & LLM Imports --
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# -- Core LLM Configuration --
LLM_GROQ = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")  # Your Google Cloud API key
MODEL = "gemini-2.0-flash-exp"
# Endpoint for Gemini (Generative Language API)
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


# -- App-specific State --
class SharedState(MessagesState):
    next_node: str = ""


class rag(BaseModel):
    is_rag: bool = False
    

# Cache the embedding model to avoid reloading it each time
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True},
    )

# Asynchronous function for transcribing audio
async def transcribe_audio_async(audio_bytes):
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Transcribe the following audio:"},
                    {"inline_data": {
                        "mime_type": "audio/wav",
                        "data": encoded_audio
                    }}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    
    # Use async version with httpx
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Extract transcript from first candidate's first part
            transcript = (
                result.get("candidates", [{}])[0]
                      .get("content", {})
                      .get("parts", [{}])[0]
                      .get("text", "")
            )
            return transcript
        else:
            return f"Error {response.status_code}: {response.text}"

# Synchronous wrapper for the async function
def transcribe_audio(audio_bytes):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(transcribe_audio_async(audio_bytes))
    loop.close()
    return result
    
# pdf file path
file_path = r"F:\AI\local\langchain_practice\genai-principles.pdf"

# Cache the document processing to avoid redoing it for each question
@st.cache_resource
def load_document(file_path, file_type):
    # Load the document based on file type
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:  # txt file
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    return text_splitter.split_documents(documents)

# -- Document Processing Function --
@st.cache_resource
def process_document(file_path, file_type):
    """Process a document and create a vector store."""
    splits = load_document(file_path, file_type)
    
    # Get cached embeddings
    dense_embeddings = get_embeddings()
    
    # Use FAISS for vector store
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=dense_embeddings
    )
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -- LangGraph Nodes --
def router_node(state: SharedState) -> SharedState:
    """Classify user input as question vs. not."""
   
    #user message
    user_msg = state["messages"][-1].content
    
    #system prompt
    sys_prompt = """
        You are an expert in routing. Your job is to determine if the user's input requires information from the document or not.
        
        ## Default Rule:
        **When in doubt, ALWAYS set `"is_rag": true`**

        ## Output Format:
        Always respond with valid JSON:
        {"is_rag": true}
        
        below is the user's message:
    """
    
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]+state["messages"]
    
    LLM_GROQ_STRUCT = LLM_GROQ.with_structured_output(rag)
    
    response = LLM_GROQ_STRUCT.invoke(messages)
    
    state["messages"].append(AIMessage(content="Routing complete"))
    state["next_node"] = "retrieve" if response.is_rag else "friendly_node"
    
    return state

def friendly_node(state: SharedState) -> SharedState:
    """Handle general conversation (not a question)."""
    
    #user message
    last_user_msg = next(msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage))

    sys_prompt ="""
    You are a friendly assistant that helps users with their documents.
    Provide a friendly, helpful response.
    If they ask for doc-specific info not yet uploaded DO NOT ANSWER, politely explain you only have 
    info from the docs they've uploaded.
    ANSWER IN THE SAME LANGUAGE AS THE USER'S MESSAGE.
    """
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=last_user_msg)] + state["messages"]
    
    response = LLM_GROQ.invoke(messages)    
    
    state["messages"].append(response)
   
    return state


def rag_node(state: SharedState) -> SharedState:
    """Answer user question using docs."""
    
    #user message
    last_user_msg = next(msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage))
    
    #content
    retriever = process_document(file_path, "pdf")
    content = retriever.invoke(last_user_msg)

    #system prompt
    sys_prompt = """
    You are a helpful assistant that answers questions based on the provided content.

    Content:
    {}

    Provide a concise answer based on the documents.
    If unsure, say: "I don't have enough information to answer this question."
    """

    messages = [SystemMessage(content=sys_prompt.format(content)), HumanMessage(content=last_user_msg)] + state["messages"]
        
    response = LLM_GROQ.invoke(messages)
    
    state["messages"].append(response)
    
    return state
    
    
def route_based_on_query_type(state: SharedState) -> str:
    return "retrieve" if state["next_node"] == "retrieve" else "friendly_node"

@st.cache_resource
def build_graph():
    memory = MemorySaver()
    workflow = StateGraph(SharedState)
    workflow.add_node("router", router_node)
    workflow.add_node("friendly_node", friendly_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_conditional_edges(
        "router",
        route_based_on_query_type,
        {
            "retrieve": "rag_node",
            "friendly_node": "friendly_node"
        }
    )
    workflow.add_edge("friendly_node", END)
    workflow.add_edge("rag_node", END)
    workflow.set_entry_point("router")
    return workflow.compile(checkpointer=memory)

# Set page configuration for a cleaner UI
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="centered"
)

# Custom CSS for a darker UI theme
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #E0E0E0;
        border: 1px solid #444444;
    }
    .stButton>button {
        background-color: #4F4F4F;
        color: #E0E0E0;
        border-radius: 5px;
        border: 1px solid #555555;
    }
    .stButton>button:hover {
        background-color: #5A5A5A;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2C333A;
    }
    .chat-message.assistant {
        background-color: #2A3129;
    }
    /* Dark scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background-color: #1E1E1E;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #4F4F4F;
        border-radius: 5px;
    }
    /* Title and text colors */
    h1, h2, h3, p, span, div {
        color: #E0E0E0 !important;
    }
    /* Container styling */
    div[data-testid="stVerticalBlock"] {
        background-color: #252525;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    /* Audio recorder button */
    button.audio-recorder {
        background-color: #4F4F4F !important;
        color: #E0E0E0 !important;
    }
    /* Input label */
    label.stTextInput {
        color: #BBBBBB !important;
    }
</style>
""", unsafe_allow_html=True)

# Set up the Streamlit page with a cleaner title
st.title("Document Assistant")
st.write("Ask questions about your document")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.graph = build_graph()

# Initialize other session state variables
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
    
if "send_message" not in st.session_state:
    st.session_state.send_message = False

# Deduplicate messages when storing
def add_message_to_history(role, content):
    # Check if this exact message already exists
    for msg in st.session_state.messages:
        if msg["role"] == role and msg["content"] == content:
            return  # Skip adding duplicate
    
    # Add new message
    st.session_state.messages.append({"role": role, "content": content})

# Function to handle message sending
def send_message():
    if st.session_state.user_text:
        # Get the current input
        user_input = st.session_state.user_text
        
        # Set flag to clear the input on next rerun
        st.session_state.send_message = True
        st.session_state.pending_message = user_input

# Callback to clear the input field after sending
if st.session_state.get("send_message", False):
    # Process the pending message
    user_input = st.session_state.pending_message
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add user message to history (deduplicating)
    add_message_to_history("user", user_input)
    
    # Process with RAG agent
    with st.spinner("Thinking..."):
        # Get response from graph
        config = {"configurable": {"thread_id": "1"}}
        response = st.session_state.graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        
        # Extract the AI's response from the result
        ai_msg = next((msg for msg in reversed(response["messages"]) 
                     if isinstance(msg, AIMessage) and "Routing complete" not in msg.content),
                     AIMessage(content="I couldn't process that request."))
        
        ai_response = ai_msg.content
        
    # Display AI response
    with st.chat_message("assistant"):
        st.write(ai_response)
    
    # Add AI response to history (deduplicating)
    add_message_to_history("assistant", ai_response)
    
    # Reset flags and transcript
    st.session_state.send_message = False
    st.session_state.transcript = ""
    st.session_state.pending_message = ""

# Display chat history in a container with scrolling
chat_container = st.container(height=50)
with chat_container:
    # Use a set to track message IDs we've already displayed
    displayed_messages = set()
    
    # Only display each unique message once
    for message in st.session_state.messages:
        # Create a tuple of (role, content) as a unique identifier
        message_id = (message["role"], message["content"])
        
        # Only display if we haven't seen this message before
        if message_id not in displayed_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
            displayed_messages.add(message_id)

# Create the input area with text and voice options side by side
input_container = st.container()
with input_container:
    col1, col2 = st.columns([5, 0.5])
    
    with col1:
        # Text input field that uses the transcript as initial value
        st.text_input(
            "Type your question here:", 
            key="user_text", 
            value=st.session_state.transcript,
            on_change=send_message
        )
    
    with col2:
        # Voice recording button
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            icon_size="2x"
        )
        
        # Process recorded audio
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(audio_bytes)
                st.session_state.transcript = transcript
                #st.rerun()  # Rerun to update the text input with transcript

# Submit button - this is an alternative to the on_change callback
if st.button("Send", use_container_width=True):
    send_message()
    st.rerun()



