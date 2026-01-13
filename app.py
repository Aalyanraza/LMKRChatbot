# Main Entry Point & FastAPI Server

from models import ChatRequest, ChatResponse, AgentState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from livekit.api import AccessToken, VideoGrants
from langchain_core.messages import AIMessage
from graph import app
import uvicorn
import config
import json
import os
# --- FastAPI Setup ---

api = FastAPI(title=config.API_TITLE)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@api.post("/chat_stream") # New Endpoint
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming endpoint that yields tokens immediately.
    """
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

    async def event_generator():
        async for event in app.astream_events(initial_state, version="v1"):
            
            # PHASE 1: STREAM TEXT (Immediate)
            if event["event"] == "on_chat_model_stream":
                if event["metadata"].get("langgraph_node") in ["generate_node", "conversational_node"]:
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            # PHASE 2: SOURCES (As soon as retrieval finishes)
            elif event["event"] == "on_chain_end":
                if event["name"] in ["retrieve_node", "career_retrieve_node", "news_retrieve_node"]:
                    output = event["data"].get("output", {})
                    chunks = output.get("context_chunks", [])
                    if chunks:
                        yield f"data: {json.dumps({'type': 'sources', 'content': chunks})}\n\n"

            # PHASE 3: VALIDATION STATUS (Async Post-Check)
            elif event["event"] == "on_chain_end":
                if event["name"] == "validate_node":
                    output = event["data"].get("output", {})
                    val_result = output.get("validation") #
                    
                    if val_result:
                        status_payload = {
                            "type": "status",
                            "is_valid": val_result.is_valid,
                            "reason": val_result.reason
                        }
                        yield f"data: {json.dumps(status_payload)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

@api.get("/get_token")
async def get_livekit_token(user_id: str = "user-1", room_name: str = "chat-room"):
    """
    Generates a secure token for the frontend to join the voice room.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="LiveKit keys missing")

    # Create a token with permissions
    grant = VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True
    )

    token = AccessToken(api_key, api_secret) \
        .with_identity(user_id) \
        .with_name(user_id) \
        .with_grants(grant)
    
    print (f"Generated LiveKit token for user {user_id} in room {room_name}")   

    return {"token": token.to_jwt(), "url": os.getenv("LIVEKIT_URL")}

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
