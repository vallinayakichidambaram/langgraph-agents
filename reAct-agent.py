from typing import Annotated, Dict, TypedDict, Sequence
from langgraph.graph import StateGraph, START,END, add_messages
from langchain_ollama import ChatOllama

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def html_context(user_query: str):
    """Use this tool to fetch relevant HTML context based on user query"""
    print("Query ", user_query)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333/",
            collection_name="html",
            embedding=embeddings,
        )
    results = vector_store.similarity_search(
        user_query, k=1
    )
    if not results:
        return "No relevant HTML context found."
    filtered_results = [doc for doc in results if doc.metadata.get("tag") not in ["comment", "link"]]

    context_text="".join("ID:"+ d.metadata["_id"] + "content" + d.page_content for d in filtered_results)
    return context_text


tools = [html_context]


llm = ChatOllama(
    base_url="http://localhost:11434/",
    model="qwen2.5-coder:1.5b",
    temperature=0
).bind_tools(tools=tools)

def call_llm(state: AgentState) -> AgentState: 
    print("Calling LLM")
    system_message = SystemMessage(content="You are a coding assistant. Based on the user query, pick the right tools to get the context and answer. If it is a new section, don't call the tools. Just write the HTML of the new section with mock data. If there is a modification in the existing section, fetch html, css and javascript (when needed) and change accordingly. Return the moified HTML, CSS and JS with ID passed in the context(if any). And write a summary about 2-3 lines on what changes have been done.")
    response = llm.invoke(
        [system_message] + state["messages"]
    )
    return {"messages": response}


def should_continue(state: AgentState):
    """This is a conditional edge. Decides it should loop or not"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "END"
    else:
        return "continue"
    

# Construct the graph
graph = StateGraph(AgentState)

graph.add_node("call_model",call_llm)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "call_model")
graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "tool_node",
        "END": END
    }
)
graph.add_edge("tool_node", "call_model")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {
    "messages": [HumanMessage(content="Change the heading in mint Section to Featured Mints")]
}

print_stream(agent.stream(inputs, stream_mode="values"))