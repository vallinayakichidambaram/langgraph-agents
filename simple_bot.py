from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from IPython.display import Image, display

class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOllama(base_url="http://localhost:11434/", model="tinyllama:1.1b")

def agent_bot(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("call_llm", agent_bot)
graph.add_edge(START, "call_llm")
graph.add_edge("call_llm", END)
agent = graph.compile()


user_input = input("Enter: ")

while(user_input != 'exit'):
    agent.invoke({
    "messages": [HumanMessage(content=user_input)]
    })
    user_input = input("Enter: ")