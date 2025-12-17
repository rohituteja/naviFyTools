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
from ollama_utils import normalize_ollama_url

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
        
        self.api_type = api_type
        self.model_name = model_name
        
        # Sanitize model name for filename
        safe_model_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_name)
        self.cache_file = f"embeddings_cache_{safe_model_name}.pkl"
        
        self.cache: dict[str, np.ndarray] = {}
        self.metadata: dict[str, object] = {
            "model_name": model_name,
            "library_size": 0
        }
        
        if api_type == "ollama":
            if not base_url:
                raise ValueError("base_url is required for Ollama")
            # Normalize Ollama URL for API calls
            self.base_url = normalize_ollama_url(base_url)
            # Store API key for Open WebUI authentication (optional for local Ollama)
            self.api_key = api_key
        else:  # openai
            if not api_key:
                raise ValueError("api_key is required for OpenAI")
            self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
            self.api_key = api_key
        
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load embeddings cache from disk with model validation."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Handle both old and new cache formats
                if isinstance(cached_data, dict):
                    # New format with metadata
                    if "metadata" in cached_data and "embeddings" in cached_data:
                        cached_model = cached_data["metadata"].get("model_name")
                        
                        # Invalidate cache if model changed
                        if cached_model != self.model_name:
                            logger.warning(f"Invalidating cache: embedding model changed from '{cached_model}' to '{self.model_name}'")
                            self.cache = {}
                            self.metadata = {
                                "model_name": self.model_name,
                                "library_size": 0
                            }
                        else:
                            self.cache = cached_data["embeddings"]
                            self.metadata = cached_data["metadata"]
                            logger.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
                    else:
                        # Old format (direct dict of embeddings)
                        logger.warning(f"Converting old cache format to new format with metadata")
                        self.cache = cached_data
                        self.metadata = {
                            "model_name": self.model_name,
                            "library_size": len(cached_data)
                        }
                        logger.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache file {self.cache_file}: {e}. Rebuilding cache.")
                self.cache = {}
                self.metadata = {
                    "model_name": self.model_name,
                    "library_size": 0
                }
                # Delete corrupted cache file
                try:
                    os.remove(self.cache_file)
                except Exception:
                    pass
        else:
            self.cache = {}
            self.metadata = {
                "model_name": self.model_name,
                "library_size": 0
            }
    
    def _save_cache(self) -> None:
        """Save embeddings cache to disk with metadata."""
        try:
            cache_data = {
                "metadata": self.metadata,
                "embeddings": self.cache
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_file}: {e}")
    
    def check_library_size(self, current_library_size: int) -> None:
        """
        Check if library size has increased and invalidate cache if needed.
        
        Args:
            current_library_size: Current number of items in the library
        """
        cached_size = self.metadata.get("library_size", 0)
        
        if current_library_size > cached_size:
            logger.warning(f"Invalidating cache: library size increased from {cached_size} to {current_library_size}")
            # Clear cache but keep some entries that are still valid
            # This is a trade-off: we could keep old embeddings, but to be safe we regenerate
            self.cache = {}
            self.metadata["library_size"] = current_library_size
            self._save_cache()
        elif current_library_size < cached_size:
            # Library shrunk, update metadata but keep cache
            logger.info(f"Library size decreased from {cached_size} to {current_library_size}, updating metadata")
            self.metadata["library_size"] = current_library_size
            self._save_cache()
        else:
            # Library size unchanged, just update the metadata count for accuracy
            self.metadata["library_size"] = current_library_size

    
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
        # Generate embedding via API
        try:
            if self.api_type == "ollama":
                url = f"{self.base_url}/embeddings"
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                # Try Ollama native format first
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=30)
                    # If 400/422, it might be Open WebUI expecting 'input'
                    if response.status_code in [400, 422]:
                        raise requests.exceptions.HTTPError(response=response)
                    response.raise_for_status()
                    data = response.json()
                except Exception:
                    # Fallback to OpenAI compatible format (Open WebUI)
                    payload["input"] = payload.pop("prompt")
                    response = requests.post(url, json=payload, headers=headers, timeout=30)
                    response.raise_for_status()
                    data = response.json()

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
            
            # Extract embedding from response (handle both Ollama and OpenAI formats)
            if "embedding" in data:
                embedding = np.array(data["embedding"], dtype=np.float32)
            elif "data" in data and len(data["data"]) > 0:
                embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            else:
                logger.warning(f"API response missing 'embedding' or 'data' field: {data}")
                return None
            
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
    
    def get_embeddings_batch(self, texts: List[str], force_refresh: bool = False) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for a list of text strings, using cache if available.
        Batches API calls where possible.
        
        Args:
            texts: List of text strings to embed
            force_refresh: If True, bypass cache and fetch new embeddings
            
        Returns:
            List of (numpy array or None) corresponding to the input texts
        """
        if not texts:
            return []
            
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # Check cache
        for i, text in enumerate(texts):
            if not text:
                continue
            if not force_refresh and text in self.cache:
                results[i] = self.cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        if not uncached_texts:
            return results

        # Process uncached items in batches
        BATCH_SIZE = 100  # Conservative batch size
        
        for i in range(0, len(uncached_texts), BATCH_SIZE):
            batch_texts = uncached_texts[i : i + BATCH_SIZE]
            batch_indices = uncached_indices[i : i + BATCH_SIZE]
            
            try:
                if self.api_type == "ollama":
                    # Try using the batch-enabled /embed endpoint first (Native Ollama)
                    url = f"{self.base_url}/embed"
                    payload = {
                        "model": self.model_name,
                        "input": batch_texts
                    }
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    
                    try:
                        response = requests.post(url, json=payload, headers=headers, timeout=60)
                        
                        # If 404, the endpoint might not exist (older Ollama or Open WebUI)
                        if response.status_code == 404:
                            raise NotImplementedError("Ollama /api/embed not found")
                        
                        response.raise_for_status()
                        data = response.json()
                        embeddings_list = data.get("embeddings", [])
                        
                        for j, embedding_data in enumerate(embeddings_list):
                            embedding = np.array(embedding_data, dtype=np.float32)
                            self.cache[batch_texts[j]] = embedding
                            results[batch_indices[j]] = embedding

                    except (requests.exceptions.RequestException, NotImplementedError):
                        # Attempt Open WebUI / OpenAI compatible batch endpoint
                        try:
                            url = f"{self.base_url}/embeddings"
                            # OpenAI style payload uses 'input' for batch
                            payload = {
                                "model": self.model_name,
                                "input": batch_texts
                            }
                            response = requests.post(url, json=payload, headers=headers, timeout=60)
                            response.raise_for_status()
                            data = response.json()
                            
                            if "data" in data:
                                for item in data["data"]:
                                    idx = item["index"]
                                    if 0 <= idx < len(batch_texts):
                                        embedding = np.array(item["embedding"], dtype=np.float32)
                                        self.cache[batch_texts[idx]] = embedding
                                        results[batch_indices[idx]] = embedding
                            else:
                                raise Exception("Missing 'data' field in batch response")
                                
                        except Exception:
                            # Fallback to serial processing
                            logger.info("Batch embedding failed for both /embed and /embeddings, falling back to serial.")
                            for j, text in enumerate(batch_texts):
                                embedding = self.get_embedding(text, force_refresh=True)
                                results[batch_indices[j]] = embedding

                else:  # openai
                    url = f"{self.base_url}/embeddings"
                    payload = {
                        "model": self.model_name,
                        "input": batch_texts
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    response = requests.post(url, json=payload, headers=headers, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    
                    # OpenAI guarantees order matches input
                    for item in data.get("data", []):
                        idx = item["index"]
                        # Robustness check: ensure index is within bounds of this batch
                        if 0 <= idx < len(batch_texts):
                            embedding = np.array(item["embedding"], dtype=np.float32)
                            self.cache[batch_texts[idx]] = embedding
                            results[batch_indices[idx]] = embedding
                            
            except Exception as e:
                logger.warning(f"Batch embedding generation failed: {e}")
        
        self._save_cache()
        return results

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
        indices = self.find_similar_indices(query, candidates, top_k)
        return [candidates[i] for i in indices]
    
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
            
            # Get embeddings for all candidates in batch
            candidate_embeddings = self.get_embeddings_batch(candidates)
            
            similarities = []
            for idx, candidate_embedding in enumerate(candidate_embeddings):
                if candidate_embedding is None:
                    continue

                # Check for dimension mismatch
                if candidate_embedding.shape != query_embedding.shape:
                    logger.warning(f"Dimension mismatch: query {query_embedding.shape} vs candidate {candidate_embedding.shape}. Skipping.")
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

