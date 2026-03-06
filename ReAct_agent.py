from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage # The foundational class for all message types in langgraph
from langchain_core.messages import ToolMessage # Passes data back to llm after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a:int, b: int) :
    """"This is an addition function that adds two numbers together"""

    return a + b

@tool
def subtract(a:int, b: int) :
    """"This is a subtraction function that subtracts two numbers"""

    return a - b

@tool
def multiply(a:int, b: int) :
    """"This is a multiplication function"""

    return a * b

tools = [add, subtract, multiply]

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="" \
    "You are my AI assistant. please answer my query to the best of your ability")

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:

        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }

)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 34 + 12 subtract 3 from 5 and multiply the result by 10 and finally tell me a Nigerian joke")]}

print_stream(app.stream(inputs, stream_mode="values"))