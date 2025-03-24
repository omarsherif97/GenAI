import os
from typing import Dict, Any, List, TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class is_log(BaseModel):
    is_log: bool = Field(..., description="Should be True if logs are present, False otherwise")

class GraphState(MessagesState):
    node: str


def helper_node(state: GraphState) -> GraphState:
    """Helper node to interact with the user."""
    prompt = """You are a helpful assistant.
        """
            
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)
    message = state["messages"]
    messages = [SystemMessage(content=prompt)] + message
   
    response = llm.invoke(messages)
    return {"messages": response}

def router_node(state: GraphState) -> Literal["LOG_ANALYZER", "HELPER"]:
    """Router node to determine which node to call next based on input."""
    # Check if the input contains log-like patterns
    message = state["messages"][-1].content
    router_prompt = "You are a router assistant. You will be given a message and you need to determine if it contains log-like patterns. If it does, return True, otherwise return False."
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)
    structured_llm = llm.with_structured_output(is_log)
    messages = [SystemMessage(content=router_prompt)] + state["messages"]
    response = structured_llm.invoke(messages)
    
    if response.is_log:
        state["node"] = "LOG_ANALYZER"
    else:
        state["node"] = "HELPER"
    return state    

def conditional_edges(state: GraphState) -> Literal["LOG_ANALYZER", "HELPER"]:
    return state["node"]


def log_analyzer(state: GraphState) -> GraphState:
    prompt = """You are a log analysis expert. Analyze the following logs and provide insights.
        
        Logs:
        {logs}
        
        Provide a detailed analysis of these logs each as a separate section,
        identifying any errors, warnings, patterns, or anomalies.
        Suggest possible causes and solutions for any issues found.
        """
    
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)
    last_user_msg = next(msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage))
    messages = [SystemMessage(content=prompt.format(logs=last_user_msg))] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}

def create_log_analyzer_graph():
    graph = StateGraph(GraphState)
    graph.add_node("LOG_ANALYZER", log_analyzer)
    graph.add_node("HELPER", helper_node)
    graph.add_node("ROUTER", router_node)
    graph.set_entry_point("ROUTER")
    
    # Use conditional edges with the router
    graph.add_conditional_edges(
        "ROUTER",
        conditional_edges,  # The router node now returns the next node name directly
        {
            "LOG_ANALYZER": "LOG_ANALYZER",
            "HELPER": "HELPER"
        }
    )
    
    graph.add_edge("LOG_ANALYZER", END)
    graph.add_edge("HELPER", END)
    return graph.compile()

# -- Expose a global variable that holds the compiled graph --
compiled_graph = create_log_analyzer_graph()


