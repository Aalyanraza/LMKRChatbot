# Main Entry Point & FastAPI Server

from fastapi import FastAPI, HTTPException
import uvicorn
from models import ChatRequest, ChatResponse, AgentState
from graph import app
import config

# --- FastAPI Setup ---

api = FastAPI(title=config.API_TITLE)

@api.get("/")
async def root():
    """
    Root endpoint - provides welcome message and usage instructions.
    """
    return {
        "message": "Welcome to the LMKR RAG Chatbot API!",
        "usage": "Use the /chat endpoint to interact.",
        "example": {
            "question": "What are the latest jobs at LMKR?",
            "user_id": "default_user"
        }
    }

@api.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # ... (initial_state setup remains the same) ...
        initial_state = {
            "question": request.question,
            "thread_id": request.user_id,
            "user_id": request.user_id,
            "retry_count": 0,
            "context_chunks": [],
            "generated_answer": None,
            "validation": None,
            "destination": "retrieve_node"
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        # 1. Extract the generated answer object
        generated_obj = result.get("generated_answer")
        
        # 2. Extract the actual source chunks from the state
        # These were populated by the retrieve_nodes
        source_chunks = result.get("context_chunks", [])
        
        # 3. Clean up the final answer string
        final_answer = "No answer generated."
        if generated_obj:
            if hasattr(generated_obj, 'answer'):
                final_answer = generated_obj.answer
            else:
                final_answer = str(generated_obj)

        # 4. Return response with both answer and source chunks
        return ChatResponse(
            answer=final_answer,
            sources=source_chunks
        )
    
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@api.get("/health")
async def health_check():
    """
    Health check endpoint - returns API status.
    """
    return {
        "status": "healthy",
        "version": "1.0",
        "service": config.API_TITLE
    }

# --- Entry Point for Debugging/Production ---

if __name__ == "__main__":
    print(f"🚀 Starting {config.API_TITLE}...")
    print(f"📡 Server running on http://{config.API_HOST}:{config.API_PORT}")
    print(f"📚 API Documentation at http://{config.API_HOST}:{config.API_PORT}/docs")
    
    uvicorn.run(
        api,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="error" # Change from default 'info' to 'error'
    )
