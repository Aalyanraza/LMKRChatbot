# Vector DB & Embeddings Setup

import torch
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import config

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDINGS_MODEL,
    model_kwargs={"device": config.EMBEDDINGS_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

embeddings._client.to(dtype=torch.bfloat16)

# Initialize Vector Store
try:
    vectorstore = FAISS.load_local(config.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.BASE_K_GENERAL})
except:
    print("⚠️ DB not found, creating dummy for execution safety.")
    vectorstore = FAISS.from_texts(["LMKR founded in 1994. GVERSE is a software brand."], embeddings)
    retriever = vectorstore.as_retriever()

# Initialize Malicious Prompt Index
malicious_embeddings = embeddings.embed_documents(config.MALICIOUS_EXAMPLES)
malicious_vectors = np.array(malicious_embeddings).astype('float32')
faiss.normalize_L2(malicious_vectors)
malicious_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
malicious_index.add(malicious_vectors)

# Initialize LLM Client
hf_client = InferenceClient(
    model=config.HF_MODEL,
    token=config.HF_API_TOKEN
)
