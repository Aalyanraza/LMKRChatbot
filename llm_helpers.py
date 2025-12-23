# LLM Helpers - Structured Querying & JSON Parsing

import json
from typing import Optional
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
import config
from embeddings_setup import hf_client

def query_llm_structured(prompt_text: str, parser: PydanticOutputParser) -> Optional[BaseModel]:
    """
    Queries the LLM with structured output enforcement.
    Handles JSON cleanup, Markdown wrapping, and Pydantic parsing.
    
    Returns parsed Pydantic object or None if parsing fails.
    """
    format_instructions = parser.get_format_instructions()
    
    final_prompt = f"""{prompt_text}

IMPORTANT INSTRUCTIONS:

1. Output ONLY a valid JSON object.

2. Do NOT output the schema definition or "properties" block. Output the actual data instance.

3. Do NOT escape underscores (e.g., use "sources_used", NOT "sources\\_used").

{format_instructions}

"""
    
    try:
        messages = [{"role": "user", "content": final_prompt}]
        response = hf_client.chat_completion(
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        
        json_str = response.choices[0].message.content.strip()
        
        # Clean Markdown wrapping
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        # Fix escaped underscores
        json_str = json_str.replace(r"\_", "_")
        
        # Detect if model returned a Schema instead of Data
        try:
            data = json.loads(json_str)
            if "properties" in data and "type" in data and data.get("type") == "object":
                print("⚠️ Model returned schema instead of data. Retrying parse...")
                return None
        except:
            pass
        
        return parser.parse(json_str)
    
    except Exception as e:
        print(f"❌ JSON Parsing/API Failed: {e}")
        return None
