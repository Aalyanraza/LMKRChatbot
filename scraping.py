import os
import re
import urllib3
from typing import List

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "Data/lmkr_data"
DATA_FILE = "lmkr_combined.txt"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

VECTOR_DB_DIR = "./vector_db"
VECTOR_DB_NAME = "faiss_lmkr"
VECTOR_DB_PATH = os.path.join(VECTOR_DB_DIR, VECTOR_DB_NAME)

# CHANGE: Using a stronger model (MPNet) for better accuracy
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# CHANGE: Smaller chunks to be more precise
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200

TARGET_URLS = [
    "https://lmkr.com/",
    "https://lmkr.com/home/company/",
    "https://lmkr.com/services-expertise/",
    "https://lmkr.com/contact/",
    "https://www.gverse.com/",
    "https://www.trverse.com/"
]

# ==========================================
# 1. SCRAPING MODULE
# ==========================================
def scrape_and_clean(urls: List[str]) -> List[Document]:
    """
    Scrapes and applies heavy cleaning to remove navigation/footers.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print(f"🕸️  Scraping {len(urls)} pages with LangChain...")
    
    loader = WebBaseLoader(
        web_paths=urls,
        verify_ssl=False,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )
    
    raw_docs = loader.load()
    cleaned_docs = []

    for doc in raw_docs:
        content = doc.page_content
        source = doc.metadata.get('source', '')
        
        # --- NOISE REMOVAL ---
        # 1. Remove common navigation/footer terms that confuse RAG
        lines = content.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines that are likely menus or irrelevant
            if len(line.strip()) < 3: continue # Skip mostly empty lines
            if any(x in line.lower() for x in ["copyright", "all rights reserved", "privacy policy", "terms of use", "skip to content"]):
                continue
            filtered_lines.append(line)
            
        content = "\n".join(filtered_lines)
        
        # 2. Normalize whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # 3. CONTEXT INJECTION
        # We prepend the source info to the text so the embedding model knows context
        doc.page_content = f"SOURCE DOCUMENT: {source}\nCONTENT:\n{content.strip()}"
        cleaned_docs.append(doc)
        print(f"   ✅ Processed: {source} ({len(doc.page_content)} chars)")

    return cleaned_docs

# ==========================================
# 2. EMBEDDING MODULE
# ==========================================
def generate_vector_db(docs: List[Document]):
    print(f"\n💎 Loading SOTA Model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True} # MPNet benefits from normalization
    )
    
    print(f"✂️  Splitting text (Chunk: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the documents
    split_docs = splitter.split_documents(docs)
    print(f"   -> Created {len(split_docs)} chunks.")

    print("💾 Building FAISS Index...")
    vector_db = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR)
        
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"✅ Saved to: {VECTOR_DB_PATH}")
    
    # Save a text backup for inspection
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        for d in split_docs:
            f.write(f"--- CHUNK FROM {d.metadata['source']} ---\n")
            f.write(d.page_content)
            f.write("\n\n")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    docs = scrape_and_clean(TARGET_URLS)
    if docs:
        generate_vector_db(docs)
        print("\n🚀 Database Updated. Now run 'debug_retrieval.py' to test it.")
    else:
        print("❌ Failed to scrape data.")