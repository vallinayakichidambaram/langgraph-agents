from typing import Dict, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(base_url="http://localhost:11434/", model="tinyllama:1.1b")

def agent_with_memory(state: AgentState) -> AgentState:
    """Use this node to call llm with conversation history"""
    result = llm.invoke(state['messages'])
    print(f"\nAI: {result.content}")
    state['messages'].append(AIMessage(content=result.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("call_llm", agent_with_memory)
graph.add_edge(START,"call_llm")
graph.add_edge("call_llm", END)

agent = graph.compile()

conversation_history = []

user_input = input("\nEnter: ")

while (user_input != 'exit'):
    conversation_history.append(HumanMessage(content=user_input))
    response = agent.invoke({
        "messages": conversation_history
    })

    conversation_history = response["messages"]
    user_input = input("\nEnter: ")