# graph.py

from langgraph.graph import StateGraph, START, END
from models import AgentState
from nodes import (
    router_node,
    retrieve_node,
    career_retrieve_node,
    news_retrieve_node,
    conversational_node,
    generate_node,
    save_memory_node,
    # input_guard_node, # Keeping import, removing from flow
    # validate_node,    # Keeping import, removing from flow
    # reflection_node   # Reflection requires validation/retries, so we skip it too
)

# --- Build the Workflow Graph ---

workflow = StateGraph(AgentState)

# Add active nodes
workflow.add_node("router_node", router_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("career_retrieve_node", career_retrieve_node)
workflow.add_node("news_retrieve_node", news_retrieve_node)
workflow.add_node("conversational_node", conversational_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("save_memory_node", save_memory_node)

# Set entry point directly to Router (Skipping Input Guard)
workflow.set_entry_point("router_node")

# Conditional edges from Router (Unchanged)
workflow.add_conditional_edges(
    "router_node",
    lambda x: x["destination"],
    {
        "career_retrieve_node": "career_retrieve_node",
        "news_retrieve_node": "news_retrieve_node",
        "retrieve_node": "retrieve_node",
        "conversational_node": "conversational_node"
    }
)

# Connect Retrieval Nodes to Generator (Unchanged)
workflow.add_edge("career_retrieve_node", "generate_node")
workflow.add_edge("news_retrieve_node", "generate_node")
workflow.add_edge("retrieve_node", "generate_node")

# Generator -> Save Memory -> END (Skipping Validation & Output Guard)
workflow.add_edge("generate_node", "save_memory_node")
workflow.add_edge("save_memory_node", END)

# Conversational Node -> END
workflow.add_edge("conversational_node", END)

# Compile the graph
app = workflow.compile()

print("✅ Linear Graph (No Guards/Validation) compiled successfully!")