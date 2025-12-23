# Pydantic Models for Structured Output

from typing import List, Optional, TypedDict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# --- Output Models for Graph Nodes ---

class QueryAugmentation(BaseModel):
    """Output for Node 1: Retrieval Augmentation"""
    augmented_queries: List[str] = Field(
        description="List of 3 alternative versions of the user question to improve search coverage."
    )

class GeneratedAnswer(BaseModel):
    """Output for Node 2: Generation"""
    answer: str = Field(description="The response to the user.")
    sources_used: List[str] = Field(default=[], description="List of context chunks or titles used.")

    @model_validator(mode='before')
    @classmethod
    def rescue_sources(cls, data):
        if isinstance(data, str):
            return {"answer": data, "sources_used": []}
        if isinstance(data, dict):
            answer_text = data.get("answer", "")
            if "Sources Used:" in answer_text and "sources_used" not in data:
                parts = answer_text.split("Sources Used:")
                data["answer"] = parts[0].strip()
                data["sources_used"] = ["Mentioned in answer"]
            return data

    @field_validator('answer', mode='before')
    @classmethod
    def flatten_list_answer(cls, v):
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return v

class ValidationResult(BaseModel):
    """Output for Node 3: Validation"""
    is_valid: bool = Field(description="True if context was used correctly and no hallucinations found.")
    reason: str = Field(description="Explanation of validation failure or success.")

class RouteDecision(BaseModel):
    """Router output model."""
    destination: Literal["career_retrieve_node", "retrieve_node", "news_retrieve_node", "conversational_node"] = Field(
        description="Choose 'news_retrieve_node' for announcements, press releases, or latest news about LMKR. Choose 'career_retrieve_node' for jobs/vacancies. Choose 'conversational_node' for chat. Choose 'retrieve_node' for everything else."
    )

# --- Agent State Definition ---

class AgentState(TypedDict):
    """State object passed through the LangGraph workflow"""
    question: str
    context_chunks: List[str]
    generated_answer: Optional[GeneratedAnswer]
    validation: Optional[ValidationResult]
    retry_count: int

# --- FastAPI Models ---

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    steps: list = Field(default_factory=list, description="Optional: show the user the reasoning steps")
