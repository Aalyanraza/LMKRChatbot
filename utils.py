# Utility Functions - Text Processing & Helpers

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def clean_text_content(text: str) -> str:
    """
    Cleans text by removing short lines and noise phrases.
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 3:
            continue
        if any(phrase in stripped.lower() for phrase in config.NOISE_PHRASES):
            continue
        cleaned_lines.append(stripped)
    
    return "\n".join(cleaned_lines)

def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)
    return chunks

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """
    Removes duplicate chunks while preserving order.
    """
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    return unique_chunks

def save_to_file(content: str, file_path: str) -> None:
    """
    Saves content to a file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def load_from_file(file_path: str) -> str:
    """
    Loads content from a file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""
