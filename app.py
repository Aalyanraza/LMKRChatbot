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
    """
    Main endpoint to interact with the RAG Chatbot.
    
    Args:
        request: ChatRequest with question and optional user_id
    
    Returns:
        ChatResponse with answer and steps
    """
    try:
        # Initialize the state
        initial_state = {
            "question": request.question,
            "retry_count": 0,
            "context_chunks": [],
            "generated_answer": None,
            "validation": None
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        # Extract the final answer
        final_answer = result.get("generated_answer", "No answer generated.")
        
        # Handle different response types
        if hasattr(final_answer, 'content'):
            final_answer = final_answer.content
        elif hasattr(final_answer, 'answer'):
            final_answer = final_answer.answer
        
        return ChatResponse(answer=str(final_answer))
    
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
        port=config.API_PORT
    )
