"""
Embedding-based optimization for naviDJ.py
Provides semantic similarity search to pre-filter large lists before sending to LLM.
"""

import os
import pickle
import logging
import requests
import numpy as np
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages embeddings with persistent caching to reduce redundant API calls.
    Supports both Ollama's /api/embeddings endpoint and OpenAI's /v1/embeddings endpoint.
    """
    
    def __init__(self, api_type: str, model_name: str, base_url: str = None, api_key: str = None, cache_file: str = "embeddings_cache.pkl"):
        """
        Initialize the EmbeddingManager.
        
        Args:
            api_type: Type of API to use ("ollama" or "openai")
            model_name: Name of the embedding model
            base_url: Base URL for Ollama (e.g., "http://localhost:11434") or OpenAI (e.g., "https://api.openai.com/v1")
            api_key: API key for OpenAI (required for OpenAI, not used for Ollama)
            cache_file: Path to the pickle file for caching embeddings
        """
        if api_type not in {"ollama", "openai"}:
            raise ValueError(f"Unsupported API type: {api_type}. Must be 'ollama' or 'openai'.")
        
            raise ValueError(f"Unsupported API type: {api_type}. Must be 'ollama' or 'openai'.")
        
        self.api_type = api_type
        self.model_name = model_name
        
        # Sanitize model name for filename
        safe_model_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_name)
        self.cache_file = f"embeddings_cache_{safe_model_name}.pkl"
        
        self.cache: dict[str, np.ndarray] = {}
        
        if api_type == "ollama":
            if not base_url:
                raise ValueError("base_url is required for Ollama")
            # Remove /v1 suffix if present for Ollama API
            self.base_url = base_url.replace("/v1", "").rstrip("/")
            self.api_key = None
        else:  # openai
            if not api_key:
                raise ValueError("api_key is required for OpenAI")
            self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
            self.api_key = api_key
        
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load embeddings cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache file {self.cache_file}: {e}. Rebuilding cache.")
                self.cache = {}
                # Delete corrupted cache file
                try:
                    os.remove(self.cache_file)
                except Exception:
                    pass
        else:
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save embeddings cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_file}: {e}")
    
    def get_embedding(self, text: str, force_refresh: bool = False) -> Optional[np.ndarray]:
        """
        Get embedding for a text string, using cache if available.
        
        Args:
            text: Text string to embed
            force_refresh: If True, bypass cache and fetch new embedding
            
        Returns:
            numpy array of embedding, or None if embedding generation fails
        """
        if not text:
            return None
        
        # Check cache first
        if not force_refresh and text in self.cache:
            return self.cache[text]
        
        # Generate embedding via API
        try:
            if self.api_type == "ollama":
                url = f"{self.base_url}/api/embeddings"
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                headers = {}
            else:  # openai
                url = f"{self.base_url}/embeddings"
                payload = {
                    "model": self.model_name,
                    "input": text
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract embedding from response (different structure for Ollama vs OpenAI)
            if self.api_type == "ollama":
                if "embedding" not in data:
                    logger.warning(f"Ollama API response missing 'embedding' field: {data}")
                    return None
                embedding = np.array(data["embedding"], dtype=np.float32)
            else:  # openai
                if "data" not in data or not data["data"]:
                    logger.warning(f"OpenAI API response missing 'data' field: {data}")
                    return None
                # OpenAI returns a list with one item containing the embedding
                embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            
            # Cache the embedding
            self.cache[text] = embedding
            self._save_cache()
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get embedding from {self.api_type}: {e}")
            if self.api_type == "ollama" and ("404" in str(e) or "model" in str(e).lower()):
                logger.warning(f"Embedding model '{self.model_name}' may not be available. Try running: ollama pull {self.model_name}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting embedding: {e}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def find_similar(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """
        Find the top_k most similar candidates to the query using cosine similarity.
        
        Args:
            query: Query string to find similar items for
            candidates: List of candidate strings
            top_k: Number of top results to return
            
        Returns:
            List of top_k most similar candidate strings
        """
        if not candidates or top_k <= 0:
            return []
        
        if len(candidates) <= top_k:
            return candidates
        
        try:
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                logger.warning("Failed to get query embedding, returning first top_k candidates")
                return candidates[:top_k]
            
            # Get embeddings for all candidates
            similarities = []
            for candidate in candidates:
                candidate_embedding = self.get_embedding(candidate)
                if candidate_embedding is None:
                    continue
                
                # Check for dimension mismatch
                if candidate_embedding.shape != query_embedding.shape:
                    logger.warning(f"Dimension mismatch: query {query_embedding.shape} vs candidate {candidate_embedding.shape}. Re-fetching...")
                    # Force refresh the embedding
                    candidate_embedding = self.get_embedding(candidate, force_refresh=True)
                    if candidate_embedding is None or candidate_embedding.shape != query_embedding.shape:
                        logger.warning(f"Could not resolve dimension mismatch for candidate. Skipping.")
                        continue

                similarity = self._cosine_similarity(query_embedding, candidate_embedding)
                similarities.append((similarity, candidate))
            
            if not similarities:
                logger.warning("No valid embeddings generated, returning first top_k candidates")
                return candidates[:top_k]
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [candidate for _, candidate in similarities[:top_k]]
            
        except Exception as e:
            logger.warning(f"Error in find_similar: {e}. Returning first top_k candidates.")
            return candidates[:top_k]
    
    def find_similar_indices(self, query: str, candidates: List[str], top_k: int) -> List[int]:
        """
        Find the top_k most similar candidates to the query, returning indices instead of strings.
        
        Args:
            query: Query string to find similar items for
            candidates: List of candidate strings
            top_k: Number of top results to return
            
        Returns:
            List of indices (0-based) of top_k most similar candidates
        """
        if not candidates or top_k <= 0:
            return []
        
        if len(candidates) <= top_k:
            return list(range(len(candidates)))
        
        try:
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                logger.warning("Failed to get query embedding, returning first top_k indices")
                return list(range(min(top_k, len(candidates))))
            
            # Get embeddings for all candidates
            similarities = []
            for idx, candidate in enumerate(candidates):
                candidate_embedding = self.get_embedding(candidate)
                if candidate_embedding is None:
                    continue

                # Check for dimension mismatch
                if candidate_embedding.shape != query_embedding.shape:
                    logger.warning(f"Dimension mismatch: query {query_embedding.shape} vs candidate {candidate_embedding.shape}. Re-fetching...")
                    # Force refresh the embedding
                    candidate_embedding = self.get_embedding(candidate, force_refresh=True)
                    if candidate_embedding is None or candidate_embedding.shape != query_embedding.shape:
                        logger.warning(f"Could not resolve dimension mismatch for candidate. Skipping.")
                        continue
                
                similarity = self._cosine_similarity(query_embedding, candidate_embedding)
                similarities.append((similarity, idx))
            
            if not similarities:
                logger.warning("No valid embeddings generated, returning first top_k indices")
                return list(range(min(top_k, len(candidates))))
            
            # Sort by similarity (descending) and return top_k indices
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [idx for _, idx in similarities[:top_k]]
            
        except Exception as e:
            logger.warning(f"Error in find_similar_indices: {e}. Returning first top_k indices.")
            return list(range(min(top_k, len(candidates))))

