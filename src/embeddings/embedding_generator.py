"""
Embedding generation for text chunks using sentence transformers.
"""
import os
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
import torch

from ..chunking.text_chunker import TextChunk
from config import config

# Try importing sentence_transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SentenceTransformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = None):
        # Use a standard sentence-transformers model instead of Azure model
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.batch_size = config.embedding.batch_size
        
        # Initialize the model
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers library not available")
            raise ImportError("SentenceTransformers library is required but not available")
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Try loading with device specification first
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model = SentenceTransformer(self.model_name, device=device)
                logger.info(f"Using {device.upper()} for embeddings")
            except Exception as device_error:
                logger.warning(f"Device-specific loading failed: {device_error}")
                # Fallback: load without device specification
                self.model = SentenceTransformer(self.model_name)
                logger.info("Loaded model without device specification")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # Final fallback: try a smaller, more compatible model
            try:
                logger.info("Attempting fallback to all-MiniLM-L6-v2...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Fallback model loading successful")
            except Exception as e2:
                logger.error(f"All model loading attempts failed: {e2}")
                raise RuntimeError(f"Could not load any embedding model. Last error: {e2}")
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with embeddings
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True if i == 0 else False
            )
            all_embeddings.extend(batch_embeddings)
        
        # Combine chunks with embeddings
        embedded_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            embedded_chunk = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'embedding': embedding,
                'metadata': chunk.metadata,
                'document_id': chunk.document_id,
                'chunk_type': chunk.chunk_type
            }
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.model.encode([query], convert_to_numpy=True)[0]
    
    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], file_path: str):
        """
        Save embeddings to disk.
        
        Args:
            embedded_chunks: Chunks with embeddings
            file_path: Path to save embeddings
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(embedded_chunks, f)
            logger.info(f"Saved embeddings to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    def load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load embeddings from disk.
        
        Args:
            file_path: Path to load embeddings from
            
        Returns:
            List of chunks with embeddings
        """
        try:
            with open(file_path, 'rb') as f:
                embedded_chunks = pickle.load(f)
            logger.info(f"Loaded embeddings from {file_path}")
            return embedded_chunks
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(config.embeddings_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        import hashlib
        content = f"{text}_{model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        cache_key = self.get_cache_key(text, model_name)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def cache_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache an embedding."""
        cache_key = self.get_cache_key(text, model_name)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")


class BatchEmbeddingProcessor:
    """Process embeddings in large batches efficiently."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.cache = EmbeddingCache()
    
    def process_document_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents and generate embeddings.
        
        Args:
            documents: List of processed documents
            
        Returns:
            List of documents with embedded chunks
        """
        all_embedded_documents = []
        
        for document in documents:
            # Import chunker here to avoid circular imports
            from src.chunking.text_chunker import SmartTextChunker, ChunkPostProcessor
            
            # Chunk the document
            chunker = SmartTextChunker()
            chunks = chunker.chunk_document(document)
            
            # Post-process chunks
            post_processor = ChunkPostProcessor()
            enhanced_chunks = post_processor.enhance_chunks(chunks)
            
            # Generate embeddings
            embedded_chunks = self.embedding_generator.generate_embeddings(enhanced_chunks)
            
            # Add embedded chunks to document
            document['embedded_chunks'] = embedded_chunks
            all_embedded_documents.append(document)
        
        return all_embedded_documents
