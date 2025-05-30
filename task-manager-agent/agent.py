from typing import Annotated, TypedDict

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str


router_llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3-groq-tool-use:8b",
    temperature=0,
    keep_alive="30m" 
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing system that determines what type of request a user makes and the requirements needed to complete the request."
            "You have access to the following tools: create_tool, delete_tool." # edit_tool
            # "If the user requests to get, see, or view one or more tasks, you will use get_tool. An example of a user request that meets this criteria is 'What are my tasks'."
            # "If the user requests to modify, change, or edit one or more tasks, you will use the edit tool. An example request that meets this criteria is 'Change my \"clean room\" task to \"clean bathroom\"'."
            "If the user requests to create, make, or add one or more tasks, you will use create_tool. Example requests that meet this critera include: 'Create a \"Clean Room\" task.', 'Create the following tasks: \"Change oil\", \"Go for a run\", and \"Check email\"'."
            "If the user requests to delete, remove, or cancel one or more tasks, you will use the delete_tool. Examples of requests that meet this criteria include 'Delete my \"Change oil\" task.', 'Delete the following tasks: \"Replace lightbuild\" and \"Eat leftovers\"'."
            "If the user requests anything unrelated to their tasks you will NOT use a tool. An example request that meets this criteria is 'Why is the sky blue?'."
        ),
        ("user", "{user_input}")
    ]
)

tasks = {"Clean Room": {"complete": False}, "Change Oil": {"complete": False}}

@tool("create_tool", parse_docstring=True)
def create(task_names: list[str]):
    """Create one or more new tasks.

    Args:
        task_names (str): The names of the tasks to create.
    """
    print(f"Tasks being created: {task_names}")
    for task_name in task_names:
        tasks[task_name] = {"complete": False}
        

@tool("delete_tool", parse_docstring=True)
def delete(task_names: list[str]):
    """Delete one or more tasks.

    Args:
        task_names (list[str]): The names of the tasks to delete.
    """
    print(f"Tasks being deleted: {task_names}")
    for task_name in task_names:
        tasks.pop(task_name)

tools = [delete, create]

def tool_router(state: State):
    if messages := state.get("messages", []):
        ai_message = messages[-1]
        assert isinstance(ai_message, AIMessage), "AIMessage required here."
    else:
        raise ValueError("Messages required here.")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    else:
        return "responder"

def responder(state: State):
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("Messages required here.")
    if isinstance(message, ToolMessage):
        tool_name = message.name
        print(f"The {tool_name} was executed.")
        if tool_name == "create_tool":
            return {"messages": ["One or more tasks were created."]}
        elif tool_name == "get_tool":
            return {"messages": ["One or more tasks were deleted."]}

    else:
        return {"messages": ["Invalid Request."]}

def router_agent(state: State):
    prompt = router_prompt.invoke({"user_input": state["user_input"]})
    return {"messages": [router_llm.invoke(prompt)]}

graph_builder = StateGraph(State)

graph_builder.add_node("router", router_agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("responder", responder)
#graph_builder.add_node("invalid_request", invalid_request)

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", tool_router, {"tools": "tools", "responder": "responder"})
graph_builder.add_edge("tools", "responder")
graph_builder.add_edge("responder", END)

graph = graph_builder.compile()