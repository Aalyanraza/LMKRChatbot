# LLM Helpers - Structured Querying using OpenAI native parsing
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from embeddings_setup import openai_client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config

# Define a TypeVar for the Pydantic model
T = TypeVar("T", bound=BaseModel)

def get_streaming_llm():
    """
    Returns a LangChain ChatOpenAI instance configured for streaming.
    This integrates with LangGraph's astream_events.
    """
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
        streaming=True 
    )

def query_llm_structured(prompt_text: str, response_model: Type[T]) -> Optional[T]:
    """
    Queries OpenAI using the Beta Parse feature. 
    This automatically enforces the Pydantic schema without manual JSON cleaning.
    """
    try:
        # We use beta.chat.completions.parse for guaranteed JSON schema adherence
        response = openai_client.beta.chat.completions.parse(
            model=config.LLM_MODEL, # Now gpt-4o-mini
            messages=[
                {"role": "system", "content": "You are a helpful corporate assistant for LMKR."},
                {"role": "user", "content": prompt_text}
            ],
            response_format=response_model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        
        return response.choices[0].message.parsed
        
    except Exception as e:
        print(f"❌ OpenAI Structured Output Failed: {e}")
        return None