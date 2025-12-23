# Vector DB & Embeddings Setup
import os
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import config

# 1. Initialize OpenAI Embeddings 
# This replaces HuggingFaceEmbeddings
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDINGS_MODEL, # "text-embedding-3-small"
    openai_api_key=config.OPENAI_API_KEY
)

# 2. Initialize Vector Store
# Note: You MUST delete your old FAISS folder and re-index. 
# 768-dim vectors will crash with OpenAI's 1536-dim embeddings.
try:
    if os.path.exists(config.VECTOR_DB_PATH):
        vectorstore = FAISS.load_local(
            config.VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": config.BASE_K_GENERAL})
        print("✅ OpenAI FAISS Index loaded successfully.")
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"⚠️ DB not found or dimension mismatch: {e}")
    print("Creating temporary store. Please run your ingestion script to rebuild the DB.")
    vectorstore = FAISS.from_texts(
        ["LMKR founded in 1994. GVERSE is a software brand."], 
        embeddings
    )
    retriever = vectorstore.as_retriever()

# 3. Initialize Malicious Prompt Index
# We re-embed your malicious examples using the new OpenAI vector space.
malicious_embeddings = embeddings.embed_documents(config.MALICIOUS_EXAMPLES)
malicious_vectors = np.array(malicious_embeddings).astype('float32')
faiss.normalize_L2(malicious_vectors)

# Create index with OpenAI dimension (1536)
malicious_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
malicious_index.add(malicious_vectors)

# 4. Initialize OpenAI Client
# This replaces HF InferenceClient for direct API access
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)