"""
PDF Processor Module
Handles PDF text extraction and intelligent text chunking for the RAG MCQ system.
"""

import re
import io
from typing import List, Dict

import PyPDF2
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def clean_text(text: str) -> str:
    """Clean extracted PDF text by removing artifacts and normalizing whitespace."""
    if not text:
        return ""
    # Remove page numbers (various formats)
    text = re.sub(r'\bPage\s*\d+\s*(of\s*\d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b-\s*\d+\s*-\b', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Remove common headers/footers patterns
    text = re.sub(r'(confidential|draft|internal use only)', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text content from an uploaded PDF file.
    
    Args:
        uploaded_file: A Streamlit UploadedFile object or any file-like object.
    
    Returns:
        Cleaned text extracted from all pages of the PDF.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        raw_text = "\n".join(text_parts)
        return clean_text(raw_text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 5, overlap: int = 2) -> List[str]:
    """Split text into overlapping chunks of sentences.
    
    Args:
        text: The full text to chunk.
        chunk_size: Number of sentences per chunk.
        overlap: Number of overlapping sentences between consecutive chunks.
    
    Returns:
        A list of text chunks, each being a group of sentences.
    """
    if not text:
        return []
    
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) <= chunk_size:
        return [text] if len(text) >= 50 else []
    
    chunks = []
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + chunk_size]
        chunk = " ".join(chunk_sentences)
        
        # Filter out chunks that are too short to be useful
        if len(chunk) >= 50:
            chunks.append(chunk)
    
    return chunks


def get_text_stats(text: str, chunks: List[str]) -> Dict:
    """Calculate statistics about the extracted text and chunks.
    
    Args:
        text: The full extracted text.
        chunks: The list of text chunks.
    
    Returns:
        A dictionary containing text and chunk statistics.
    """
    sentences = nltk.sent_tokenize(text) if text else []
    words = text.split() if text else []
    avg_chunk_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    
    return {
        'total_characters': len(text),
        'total_words': len(words),
        'total_sentences': len(sentences),
        'total_chunks': len(chunks),
        'avg_chunk_length': round(avg_chunk_len, 1)
    }
