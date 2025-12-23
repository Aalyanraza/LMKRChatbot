# Graph Construction & Workflow Compilation
# from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from models import AgentState
from nodes import (
    input_guard_node,
    router_node,
    retrieve_node,
    career_retrieve_node,
    news_retrieve_node,
    conversational_node,
    generate_node,
    output_guard_node,
    validate_node
)

# --- Edge Routing Logic ---

def validation_router(state: AgentState):
    """
    Determines where to route based on validation results.
    Returns END if valid or max retries reached, otherwise loops back to appropriate node.
    """
    validation = state.get("validation")
    retry_count = state.get("retry_count", 0)
    destination = state.get("destination", "retrieve_node")
    
    # 1. Success
    if validation and validation.is_valid:
        print("✅ Validation Passed.")
        return END
    
    # 2. Max Retries
    if retry_count >= 2:
        print("🛑 Max retries reached. Returning best effort.")
        return END
    
    # 3. FAILURE -> LOOP BACK
    print(f"🔄 Validation Failed: {validation.reason if validation else 'Unknown'}. Expanding search context...")
    
    if destination == "career_retrieve_node":
        return "career_retrieve_node"
    elif destination == "news_retrieve_node":
        return "news_retrieve_node"
    elif destination == "conversational_node":
        return "conversational_node"
    else:
        return "retrieve_node"

# --- Build the Workflow Graph ---

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("input_guard_node", input_guard_node)
workflow.add_node("router_node", router_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("career_retrieve_node", career_retrieve_node)
workflow.add_node("news_retrieve_node", news_retrieve_node)
workflow.add_node("conversational_node", conversational_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("output_guard_node", output_guard_node)
workflow.add_node("validate_node", validate_node)

# Set entry point
workflow.set_entry_point("input_guard_node")

# Input Guard -> Router
workflow.add_edge("input_guard_node", "router_node")

# Conditional edges from Router
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

# Connect Retrieval Nodes to Generator
workflow.add_edge("career_retrieve_node", "generate_node")
workflow.add_edge("news_retrieve_node", "generate_node")
workflow.add_edge("retrieve_node", "generate_node")

# Generator -> Output Guard
workflow.add_edge("generate_node", "output_guard_node")

# Output Guard -> Validator
workflow.add_edge("output_guard_node", "validate_node")

# Conditional edges from Validator (The Loop)
workflow.add_conditional_edges(
    "validate_node",
    validation_router,
    {
        END: END,
        "retrieve_node": "retrieve_node",
        "career_retrieve_node": "career_retrieve_node",
        "news_retrieve_node": "news_retrieve_node",
        "conversational_node": "conversational_node"
    }
)

# Conversational Node -> END
workflow.add_edge("conversational_node", END)

# Compile the graph
app = workflow.compile()

# visulization (optional)
# graph_repr = app.get_graph() 
# png_bytes = graph_repr.draw_mermaid_png() 
# with open("workflow_graph.png", "wb") as f:
#     f.write(png_bytes)

print("✅ Graph compiled successfully!")
