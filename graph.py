# graph.py

from langgraph.graph import StateGraph, START, END
from models import AgentState
from nodes import (
    decision_node,
    tool_execution_node,
    generate_node,
    save_memory_node,
    # input_guard_node # (Optional, if you want to use it)
)

# --- Build the Workflow Graph ---

workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("decision_node", decision_node)
workflow.add_node("tool_execution_node", tool_execution_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("save_memory_node", save_memory_node)

# 2. Set Entry Point
workflow.set_entry_point("decision_node")

# 3. Define Conditional Logic
def route_decision(state):
    if state.get("tool_calls") and len(state["tool_calls"]) > 0:
        return "tool_execution_node"
    return "generate_node"

# 4. Add Conditional Edge
workflow.add_conditional_edges(
    "decision_node",
    route_decision,
    {
        "tool_execution_node": "tool_execution_node",
        "generate_node": "generate_node"
    }
)

# 5. Connect Tool Execution to Generator
# (Once tools run, we ALWAYS generate an answer)
workflow.add_edge("tool_execution_node", "generate_node")

# 6. Connect Generator to Memory/End
workflow.add_edge("generate_node", "save_memory_node")
workflow.add_edge("save_memory_node", END)

# Compile
app = workflow.compile()

print("✅ Agentic Tool Graph (Decision -> [Tools] -> Generate) compiled!")