import os
import re
import requests
from typing import List
from bs4 import BeautifulSoup
import faiss
import numpy as np

# LangChain Imports
from langchain_openai import OpenAIEmbeddings # Changed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import config


# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "Data/lmkr_data"
DATA_FILE = "lmkr_combined.txt"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

VECTOR_DB_DIR = "./vector_db"
VECTOR_DB_NAME = "faiss_lmkr"
VECTOR_DB_PATH = os.path.join(VECTOR_DB_DIR, VECTOR_DB_NAME)

EMBEDDING_MODEL_NAME = config.EMBEDDINGS_MODEL
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TARGET_URLS = [
    "https://lmkr.com/",
    "https://lmkr.com/home/company/",
    "https://lmkr.com/services-expertise/",
    "https://lmkr.com/contact/",
    "https://www.gverse.com/about",
    "https://www.gverse.com/solutions",
    "https://www.gverse.com/solutions/geology-solutions",
    "https://www.gverse.com/solutions/data-management-solutions",
    "https://www.gverse.com/solutions/value-added-geoscience-services",
    "https://www.gverse.com/solutions/geophysics-solutions",
    "https://www.gverse.com/solutions/field-development-solutions",
    "https://trverse.com/",
    "https://trverse.com/who-we-are/",
    "https://trverse.com/our-mission/",
    "https://trverse.com/our-vision/",
    "https://trverse.com/transport-management/",
    "https://trverse.com/information-system/",
    "https://trverse.com/integrated-security/",
    "https://trverse.com/automated-fare-collection/",
    "https://trverse.com/intelligent-transport-system/",
    "https://trverse.com/signal-priority-system/"
]

# ==========================================
# 1. SCRAPING MODULE
# ==========================================

def get_soup(url: str):
    """Helper to get BeautifulSoup object safely."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def create_global_context_doc(main_url: str) -> Document:
    """
    Scrapes the Header and Footer ONLY for the global context.
    """
    print(f"🌍 Extracting Global Context (Header/Footer) from {main_url}...")
    soup = get_soup(main_url)
    if not soup: return None

    global_text = []
    
    # Grab Footer
    footer = soup.find('footer')
    if footer:
        global_text.append("=== COMPANY CONTACT & FOOTER INFO ===")
        global_text.append(footer.get_text(separator="\n").strip())

    # Grab Nav
    nav = soup.find('nav')
    if nav:
        global_text.append("=== SITE NAVIGATION STRUCTURE ===")
        global_text.append(nav.get_text(separator=" | ").strip())

    full_text = "\n\n".join(global_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return Document(
        page_content=full_text, 
        metadata={"source": "GLOBAL_CONTEXT_HEADER_FOOTER"}
    )

def fetch_and_clean_body(url: str) -> str:
    """
    Scrapes the page but preserves content even if it's in a <form> or <header>.
    """
    soup = get_soup(url)
    if not soup: return ""

    # 1. REMOVE ONLY NON-CONTENT TAGS
    # REMOVED 'form' and 'header' from this list because some sites use them 
    # to wrap the main content or hero sections.
    tags_to_remove = ["nav", "footer", "script", "style", "noscript", "iframe", "svg"]
    
    for tag in soup(tags_to_remove):
        tag.decompose()

    # 2. REMOVE NOISE BY CLASS
    # We are careful not to remove 'widget' blindly as some page builders use it for content
    noise_classes = ["cookie", "popup", "newsletter", "signup", "login", "breadcrumb", "sidebar"]
    for noise in noise_classes:
        for element in soup.find_all(class_=re.compile(noise, re.IGNORECASE)):
            element.decompose()

    # 3. EXTRACTION STRATEGY
    # First, try to find the "main" content area to avoid grabbing menu leftovers
    main_content = soup.find('main') or soup.find('article') or soup.find(id='content')
    
    if main_content:
        text = main_content.get_text(separator="\n")
    else:
        # Fallback: Get everything remaining in the body
        text = soup.get_text(separator="\n")

    return text

def clean_text_content(text: str, source: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    
    NOISE_PHRASES = [
        "warning", "required", "page load link", "skip to content", 
        "please click here", "redirected", "all rights reserved", 
        "enable javascript", "browser not supported"
    ]

    for line in lines:
        stripped = line.strip()
        
        # Keep short lines if they look like headers (no symbols)
        if len(stripped) < 3: 
            continue
        
        # Check for noise phrases
        if any(phrase in stripped.lower() for phrase in NOISE_PHRASES):
            continue
            
        cleaned_lines.append(stripped)

    final_text = "\n".join(cleaned_lines)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)
    
    # Debug: Warn if content is empty (Possible JavaScript site)
    if len(final_text) < 50:
        print(f"⚠️  WARNING: Content for {source} is surprisingly short. The site might require JavaScript.")

    return f"SOURCE DOCUMENT: {source}\n{final_text}"

def scrape_urls(urls: List[str]) -> List[Document]:
    documents = []
    
    # 1. Global Context
    global_doc = create_global_context_doc(urls[0])
    if global_doc:
        documents.append(global_doc)
        print("   ✅ Added Global Header/Footer Context")

    # 2. Scrape Pages
    print(f"🕸️  Scraping {len(urls)} pages...")
    for url in urls:
        raw_text = fetch_and_clean_body(url)
        if raw_text:
            clean_content = clean_text_content(raw_text, url)
            doc = Document(page_content=clean_content, metadata={"source": url})
            documents.append(doc)
            print(f"   ✅ Processed: {url} ({len(clean_content)} chars)")
        else:
            print(f"   ❌ Failed to extract text: {url}")
            
    return documents

# ==========================================
# 2. EMBEDDING MODULE
# ==========================================
def generate_vector_db(docs: List[Document]):
    if not docs: 
        return

    print(f"\n💎 Initializing OpenAI Embeddings: {EMBEDDING_MODEL_NAME}...")
    # Switched to OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    print(f"✂️  Splitting text into {CHUNK_SIZE} char chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    split_docs = splitter.split_documents(docs)
    
    # 1. Embed the documents
    print(f"🧠 Embedding {len(split_docs)} chunks...")
    doc_texts = [d.page_content for d in split_docs]
    vectors = np.array(embeddings.embed_documents(doc_texts)).astype('float32')
    
    # 2. Setup the Quantized Index
    # IMPORTANT: Dimension must be 1536 for OpenAI 3-small
    dimension = 1536 
    index = faiss.index_factory(dimension, "SQ8") 
    
    # 3. Train and Build
    print("🎓 Training and Building Quantized FAISS Index...")
    index.train(vectors)
    
    vector_db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    
    vector_db.add_documents(split_docs)
    
    # 5. Save
    if not os.path.exists(config.VECTOR_DB_DIR): os.makedirs(config.VECTOR_DB_DIR)
    vector_db.save_local(config.VECTOR_DB_PATH)
    print(f"✅ OpenAI FAISS database saved to {config.VECTOR_DB_PATH}")



if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    docs = scrape_urls(TARGET_URLS)
    generate_vector_db(docs)