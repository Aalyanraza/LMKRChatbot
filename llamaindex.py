import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
# Import LlamaIndex components for custom LLM
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any, List, Optional
# LlamaIndex decorator for observability
from llama_index.core.llms.callbacks import llm_completion_callback 

# --- 0. Setup Environment and HF Client ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

if HF_API_TOKEN is None:
    raise RuntimeError(
        "HF_API_TOKEN environment variable is not set."
    )

hf_client = InferenceClient(
    model=HF_MODEL_ID,
    token=HF_API_TOKEN,
    timeout=60,
)

def hf_generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    # Use your existing, robust generation logic
    try:
        completion = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        msg = completion.choices[0].message
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        
        return content or ""
    except Exception as e:
        print(f"❌ HF API Error: {e}")
        return "I don't have this information."

# --- 1. Custom LLM Wrapper for LlamaIndex ---
class HuggingFaceCustomLLM(CustomLLM):
    """
    A LlamaIndex LLM wrapper for the HuggingFace Inference Client.
    """
    # Define metadata required by LlamaIndex
    context_window: int = 4096 
    num_output: int = 512
    model_name: str = HF_MODEL_ID

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    # Implement the synchronous completion method
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Pass the LlamaIndex prompt directly to your generation function
        response_text = hf_generate(prompt, max_new_tokens=self.num_output)
        return CompletionResponse(text=response_text)

    # NOTE: You must also implement the streaming method for LlamaIndex,
    # but we'll leave it as a placeholder to keep this example simple and working.
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Streaming is not implemented for this example.")

# --- 2. LlamaIndex RAG Pipeline ---

# Set the Custom LLM globally
Settings.llm = HuggingFaceCustomLLM() 

# Set the Local Embedding Model globally
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load data (assuming 'data' directory/file path is now correct)
print("Loading and Indexing data...")
documents = SimpleDirectoryReader("data").load_data()

# Index the data (Uses Settings.embed_model)
index = VectorStoreIndex.from_documents(documents)

# Create Query Engine (Uses Settings.llm)
query_engine = index.as_query_engine()

# Start the chat loop
print("✅ RAG Engine Ready. Ask your question.")
question = ""
while (question.lower() != "exit"):
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() != "exit":
        response = query_engine.query(question)
        print("Response:", response)