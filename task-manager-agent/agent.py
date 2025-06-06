from typing import Annotated, TypedDict

from langchain_core.tools.base import InjectedToolCallId
from langchain_core.runnables import Runnable
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import numpy as np

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt, Send
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

import numpy.typing as npt

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

out = load_dotenv(".env")
print(out)

try:
    ssm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
except Exception:
    ssm = SentenceTransformer("/models/all-minilm", device="cpu")
    ssm.save_pretrained("/models/all-minilm")


def pairwise_cosine_similarity(a: npt.NDArray, b: npt.NDArray):
    """Calculate the pairwise cosine similarity.

    Args:
        a (npt.NDArray): A numpy array of one or more embeddings.
        b (npt.NDArray): A numpy array of one or more embeddings.

    Returns:
        npt.NDArray: The similarily scores. If input `b` is a matrix the returned shape is (1, k), where k is the number of rows in matrix `b`. If both `a` and `b` are matrices then the returned shape is (h, k).
    """
    assert a.ndim >= 1 and b.ndim == 2, (
        f"Input 'a' has ({a.ndim}), and 'b' has ({b.ndim})."
    )
    return a @ b.transpose(1, 0)


def encode(task_names: str | list[str]) -> npt.NDArray:
    """Normalized Embeddings of task names.

    Args:
        task_names (str | list[str]): A task name, or names.

    Returns:
        npt.NDArray: The embedding(s).
    """
    return ssm.encode(task_names, convert_to_numpy=True, normalize_embeddings=True)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str


router_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing system that determines what type of request a user makes and the requirements needed to complete the request."
            "You have access to the following tools: create_task, edit_task, delete_task."  # edit_tool
            # "If the user requests to get, see, or view one or more tasks, you will use get_tool. An example of a user request that meets this criteria is 'What are my tasks'."
            # "If the user requests to modify, change, or edit one or more tasks, you will use the edit tool. An example request that meets this criteria is 'Change my \"clean room\" task to \"clean bathroom\"'."
            "If the user requests to create, make, or add one or more tasks, you will use the create_task tool. Example requests that meet this critera include: 'Make a Clean Room task.' and 'Create the following tasks: Change oil, Go for a run, and Check email.'."
            "If the user requests to edit, modify, or change one of their tasks, you will use the edit_task tool. Examples include: 'Mark my do homework task as complete' and 'Change the name of my clean room task to clean bathroom.'."
            "If the user requests to edit, modify, or change multiple tasks you will use the edit_tasks tool. Examples include: 'Mark my mow the lawn and do the laundry tasks as complete.' and 'Change the name of my get groceries task to get fruits and vegetables, and change the name of my do homework task to do abstract algebra homework.'."
            "If the user requests to delete, remove, or cancel one of their tasks, you will use the delete_task tool. Examples of requests that meet this criteria include: 'Delete my change oil task.', 'Discard the following task, replace lightbuld.'."
            "If the user request to delete, remove, cancel, etc. several of their tasks you will use the delete_tasks tool. An example that meet this criteria is: 'Delete the change baby's diapers task and the do calculus homework task.'."
            "If the user requests anything unrelated to their tasks you will NOT use a tool. An example request that meets this criteria is 'Why is the sky blue?'."
            "The user's current tasks are: {tasks}.",
        ),
        ("user", "{user_input}"),
    ]
)

CANCEL_EMB = np.expand_dims(encode("Cancel request."), axis=0)


class Task(TypedDict):
    name: str
    is_complete: bool
    embed: npt.NDArray


class TaskManager:
    def __init__(self, tasks: list | None = None):
        self.lut = {}
        if tasks is not None:
            self.lut.update({t["name"]: t for t in tasks})

    def display_tasks(self) -> str:
        return "'" + "', '".join(self.lut.keys()) + "'"

    # Replace with in-memory vector store
    def get_embeds(self) -> tuple[npt.NDArray, list[str]] | None:
        embs = []
        names = []
        for k, v in self.lut.items():
            embs.append(v["embed"])
            names.append(k)
        if len(embs) > 0 and len(names) > 0:
            return np.stack(embs), names
        else:
            return None

    def create_task(
        self,
        task_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        is_complete: bool = False,
    ):
        embed: npt.NDArray = encode(task_name)
        self.lut[task_name] = Task(name=task_name, is_complete=is_complete, embed=embed)

    def create_tasks(
        self,
        task_names: list[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
        are_complete: list[bool] | None = None,
    ):
        if are_complete is None:
            are_complete = [False] * len(task_names)
        embeds = encode(task_names)
        for task_name, is_complete, embed in zip(task_names, are_complete, embeds):
            self.lut[task_name] = Task(
                name=task_name, is_complete=is_complete, embed=embed
            )

    def edit_task(
        self,
        old_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        new_name: str | None,
        is_complete: bool | None,
    ):
        embs_and_names = self.get_embeds()
        if embs_and_names is None:
            return ToolMessage(
                "No tasks exist, so editing one is not possible.",
                tool_call_id=tool_call_id,
            )
        else:
            embs, names = embs_and_names
            old_name_emb = encode(old_name)
            sim_scores = pairwise_cosine_similarity(old_name_emb, embs)
            print(f"Similarity Scores: {sim_scores}")
            most_similar_idx: np.int64 = np.argmax(sim_scores)
            if sim_scores[most_similar_idx] < 0.85:
                human_response = interrupt(
                    "Unable to match your request to an existing task. Please provide more info or say 'cancel request'."
                )
                if (
                    pairwise_cosine_similarity(
                        encode(human_response), CANCEL_EMB
                    ).item()
                    > 0.85
                ):
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    "Your previous request has been cancelled",
                                    tool_call_id=tool_call_id,
                                )
                            ]
                        }
                    )
                else:
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    "Tool call was unsuccessful.",
                                    tool_call_id=tool_call_id,
                                )
                            ],
                            "user_input": human_response,
                        },
                        goto="router_agent",
                    )
            else:
                name = names[most_similar_idx]
                if new_name is not None:
                    self.lut[new_name] = self.lut.pop(name)
                    self.lut[new_name]["name"] = new_name
                    self.lut[new_name]["embed"] = encode(new_name)

                if is_complete is not None:
                    self.lut[name]["is_complete"] = is_complete

                # Use additional `if` statements for other task attributes

    def delete_task(self, task_name: str, tool_call_id: str):
        embs_and_names = self.get_embeds()
        if embs_and_names is None:
            return ToolMessage(
                "No tasks exist, so deletion is not possible.",
                tool_call_id=tool_call_id,
            )
        else:
            embs, names = embs_and_names
            new_emb = encode(task_name)
            sims = pairwise_cosine_similarity(new_emb, embs)
            print(f"Similarity Scores: {sims}")
            most_similar_idx: np.int64 = np.argmax(sims)
            if sims[most_similar_idx] < 0.85:
                human_response = interrupt(
                    "Unable to match your deletetion request to an existing task. Please provide more info or say 'cancel request'."
                )
                if (
                    pairwise_cosine_similarity(
                        encode(human_response), CANCEL_EMB
                    ).item()
                    > 0.85
                ):
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    "Your previous request has been cancelled",
                                    tool_call_id=tool_call_id,
                                )
                            ]
                        }
                    )
                else:
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    "Tool call was unsuccessful.",
                                    tool_call_id=tool_call_id,
                                )
                            ],
                            "user_input": human_response,
                        },
                        goto="router_agent",
                    )
            name = names[most_similar_idx]
            self.lut.pop(name)

    def delete_tasks(self, task_names: list[str], tool_call_id: str):
        embs_and_names = self.get_embeds()
        if embs_and_names is None:
            return ToolMessage(
                "No tasks exist, so deletion is not possible.",
                tool_call_id=tool_call_id,
            )
        else:
            cur_embs, names = embs_and_names
            new_embs = encode(task_names)
            sim_scores = pairwise_cosine_similarity(new_embs, cur_embs)
            most_similar_idxs: npt.NDArray[np.int64] = np.argmax(sim_scores, axis=1)
            # Check that each 'most_similar_idx' corresponds to a similarity score higher than the threshold
            no_match = []  # task names that don't have a high enough similarity score
            for i, most_similar_idx in enumerate(most_similar_idxs):
                if sim_scores[i, most_similar_idx] < 0.85:
                    no_match.append(i)
                else:
                    try:
                        self.lut.pop(names[most_similar_idx])
                    except KeyError:
                        print(
                            f"The task matching ({names[most_similar_idx]}) has already been matched and deleted."
                        )
                        pass
            if len(no_match) > 0:
                human_response = interrupt(
                    "Unable to delete the following tasks: "
                    + ", ".join([task_names[i] for i in no_match])
                    + ". Please provide more info or say 'cancel request'."
                )
                if (
                    pairwise_cosine_similarity(
                        encode(human_response), CANCEL_EMB
                    ).item()
                    > 0.80
                ):
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    "Your previous request has been cancelled",
                                    tool_call_id=tool_call_id,
                                )
                            ]
                        }
                    )
                else:
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    "Tool call was unsuccessful.",
                                    tool_call_id=tool_call_id,
                                )
                            ],
                            "user_input": human_response,
                        },
                        goto="router_agent",
                    )


tasks = [
    {"name": "Clean Room", "is_complete": False, "embed": encode("Clean Room")},
    {"name": "Change Oil", "is_complete": False, "embed": encode("Change Oil")},
]

task_manager = TaskManager(tasks)


@tool("create_task", parse_docstring=True)
def create_task(task_name: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Create (i.e. add, or make) a task.

    Args:
        task_names (str): The name of the task to create.
    """
    task_manager.create_task(task_name, tool_call_id=tool_call_id)
    return f"The {task_name} task was created."


@tool("create_tasks", parse_docstring=True)
def create_tasks(
    task_names: list[str], tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Create (i.e. add or make) multiple tasks.

    Args:
        task_names (list[str]): The names of the tasks to create.
        tool_call_id (Annotated[str, InjectedToolCallId]): _description_

    Returns:
        _type_: _description_
    """
    task_manager.create_tasks(task_names, tool_call_id=tool_call_id)
    return "The following tasks were created: " + ", ".join(task_names) + "."


@tool("delete_task", parse_docstring=True)
def delete_task(task_name: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Delete (i.e. remove or cancel) a single task.

    Args:
        task_name (str): The task name.
        tool_call_id (Annotated[str, InjectedToolCallId]):
    """
    print(f"The tool 'delete_task' was called on the task ({task_name}).")
    output = task_manager.delete_task(task_name, tool_call_id)
    if output is not None:
        return output
    else:
        return f"The task '{task_name}' was successfully deleted."


@tool("delete_tasks", parse_docstring=True)
def delete_tasks(
    task_names: list[str], tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Delete (i.e. remove, or cancel) a list of tasks.

    Args:
        task_names (str): The names of the tasks to delete.
    """
    output = task_manager.delete_tasks(task_names, tool_call_id)
    if output is not None:
        return output
    tasks_as_str = ", ".join(task_names)
    # raise NotImplementedError("You may only delete one task at a time!")
    return "The following tasks were deleted: " + tasks_as_str


@tool("edit_task", parse_docstring=True)
def edit(
    task_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    new_name: str | None = None,
    is_complete: bool | None = None,
):
    task_manager.edit_task(task_name, tool_call_id, new_name, is_complete)
    return f"The {task_name} task was edited."


tools = [delete_task, delete_tasks, create_task]
router_llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3-groq-tool-use:8b-q3_K_M",  # "llama3-groq-tool-use:8b",
    temperature=0,
    keep_alive="30m",
).bind_tools(tools)

router_llm.invoke("")  # dummy call to load model on gpu

validator_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        (
            "system",
            "You are a security and validation system. You analyze text and determine if it relates to 'task managment': specifically, you verify that a user's request relates to creating tasks, editing tasks, or deleting tasks. You classify all other input as invalid.",
        ),
        ("{user_input}"),
    ]
)

# NOTE: A more optimal solution might utilize a fine-tuned SSM to validate user requests.
validator_llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:3B",
    temperature=0,
    keep_alive="30m",
)

class Validation(BaseModel):
    is_valid_request: bool = Field(default=..., description="A boolean value indicating if the user's input is valid and relevant to task management.")

validator_chain: Runnable = validator_template | validator_llm.with_structured_output(Validation)


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


# TODO: This should not be a HumanMessage (probably): look into it!
def responder(state: State):
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("The messages attribute is required here.")
    if isinstance(message, ToolMessage):
        tool_name = message.name
        assert message.content != "null"
        tool_msg = message.content
        print(f"The {tool_name} was executed.")
        if tool_name == "create_tool":
            return {"messages": [tool_msg]}
        elif tool_name == "get_tool":
            return {"messages": [tool_msg]}
    else:
        return {"messages": ["Invalid Request."]}


def routing_agent(state: State):
    assert isinstance(state, dict), (
        "Input to router_agent node must be of the State class (i.e. a typed dict)"
    )
    prompt = router_agent_prompt.invoke(
        {"user_input": state["user_input"], "tasks": task_manager.display_tasks()}
    )
    return {"messages": [router_llm.invoke(prompt)]}

def validation_agent(state: State):
    user_input = state["user_input"]
    assert isinstance(user_input, str) and len(user_input) > 0
    return {"messages": [validator_chain.invoke(user_input)]}

graph_builder = StateGraph(State)

graph_builder.add_node("validation_agent", validation_agent)
graph_builder.add_node("routing_agent", routing_agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("responder", responder)
# graph_builder.add_node("invalid_request", invalid_request)

graph_builder.add_edge(START, "router_agent")
graph_builder.add_conditional_edges(
    "router_agent", tool_router, {"tools": "tools", "responder": "responder"}
)
graph_builder.add_edge("tools", "responder")
graph_builder.add_edge("responder", END)

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)
