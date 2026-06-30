"""
RAG Retriever Module
Implements TF-IDF based information retrieval with cosine similarity
for context-aware chunk retrieval.
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    """TF-IDF based retriever for finding relevant text chunks.
    
    This is the core RAG component. It converts text chunks into TF-IDF vectors
    and retrieves the most relevant ones based on cosine similarity with a query.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Initialize the retriever.
        
        Args:
            max_features: Maximum number of features for TF-IDF.
            ngram_range: Range of n-grams to consider.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.chunks = []
        self.is_fitted = False
    
    def fit(self, chunks: List[str]) -> 'TFIDFRetriever':
        """Fit the retriever on a list of text chunks.
        
        Args:
            chunks: List of text passages to index.
        
        Returns:
            Self, for method chaining.
        """
        if not chunks:
            raise ValueError("Cannot fit retriever on empty chunks list.")
        
        self.chunks = chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        self.is_fitted = True
        return self
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the most relevant chunks for a given query.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
        
        Returns:
            List of dicts with 'chunk', 'score', and 'index' keys.
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever must be fitted before retrieval. Call fit() first.")
        
        if not query.strip():
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top_k indices sorted by similarity (descending)
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include chunks with some relevance
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def retrieve_for_difficulty(self, query: str, difficulty: str, num_chunks: int = 5) -> List[Dict]:
        """Retrieve chunks with difficulty-adjusted breadth.
        
        Args:
            query: Search query string.
            difficulty: One of 'easy', 'medium', 'hard'.
            num_chunks: Base number of chunks to retrieve.
        
        Returns:
            List of relevant chunk dicts.
        """
        difficulty_map = {
            'easy': num_chunks,
            'medium': num_chunks + 3,
            'hard': num_chunks + 5
        }
        top_k = difficulty_map.get(difficulty.lower(), num_chunks)
        return self.retrieve(query, top_k=top_k)
    
    def get_chunk_by_index(self, index: int) -> Optional[str]:
        """Get a specific chunk by its index.
        
        Args:
            index: The index of the chunk.
        
        Returns:
            The chunk text, or None if index is invalid.
        """
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract top keywords from text using TF-IDF scoring.
    
    Args:
        text: Input text to extract keywords from.
        top_n: Number of top keywords to return.
    
    Returns:
        List of keywords sorted by TF-IDF importance.
    """
    if not text.strip():
        return []
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    
    keyword_scores = list(zip(feature_names, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [kw for kw, score in keyword_scores[:top_n]]
