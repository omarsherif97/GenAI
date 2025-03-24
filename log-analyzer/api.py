from fastapi import FastAPI
from log_analyzer import compiled_graph, GraphState
from langchain_core.messages import HumanMessage
from gradio_utils import demo
import gradio.routes as gr_routes

# Create an instance of the FastAPI class
app = FastAPI()

# Define a basic GET endpoint at the root URL ("/")
@app.post("/")
async def graph_endpoint(log_data: str):
    result = await compiled_graph.ainvoke({"messages": [HumanMessage(content=log_data)]})
    return result.get("messages", [])[-1].content

# Define a GET endpoint with a path parameter
@app.get("/users/{user_id}")
def read_user(user_id: int):
    return {"user_id": user_id, "info": "Here is some user info"}

# Mount the Gradio app
app = gr_routes.mount_gradio_app(app, demo, path="/gradio")