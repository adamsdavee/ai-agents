from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage # The foundational class for all message types in langgraph
from langchain_core.messages import ToolMessage # Passes data back to llm after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage, HumanMessage # Message for providing instructions to the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os


load_dotenv()

# This is a global variable to store document content
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """"Updates the document with the provided content"""

    global document_content
    document_content = content

    return f"Document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """"Save the current document to a text file and finish the process

    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved to: {filename}")

        return f"Document has been saved successfully to '{filename}'"
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]


llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.

        The current document content is:{document_content}
        """)
    
    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create"
        user_message = HumanMessage(content=user_input)

    else: 
        user_input = "\nWhat would you like to do with the document? "
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)
