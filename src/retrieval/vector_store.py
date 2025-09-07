"""
Vector database integration for semantic search using FAISS and Pinecone.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import faiss
from loguru import logger

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available. Install pinecone-client for Pinecone support.")

from config import config


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors with metadata to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for local semantic search."""
    
    def __init__(self, dimension: int = None):
        self.dimension = dimension or 384  # Use 384 for all-MiniLM-L6-v2
        self.index = None
        self.metadata = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS index with dimension {self.dimension}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors with metadata to FAISS index."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, k: int = 5, document_filter: List[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors in FAISS index with optional document filtering."""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search with more results if filtering is needed - cast a wider net
        search_k = k * 5 if document_filter else k * 2
        scores, indices = self.index.search(query_vector, min(search_k, self.index.ntotal))
        
        # Prepare results with optional document filtering - be more permissive with similarity
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                metadata = self.metadata[idx]
                
                # Apply document filter if specified
                if document_filter:
                    source_file = metadata.get('source_file', '')
                    if not any(doc_name in source_file for doc_name in document_filter):
                        continue
                
                # Include results with lower similarity threshold for better coverage
                if float(score) > 0.1:  # Very permissive threshold
                    results.append((metadata, float(score)))
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        return results
    
    def save(self, path: str) -> None:
        """Save FAISS index and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str) -> None:
        """Load FAISS index and metadata."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata
        with open(f"{path}.metadata", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded FAISS index from {path}")


class PineconeVectorStore(VectorStore):
    """Pinecone-based vector store for cloud semantic search."""
    
    def __init__(self, index_name: str = None, api_key: str = None, environment: str = None):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install pinecone-client.")
        
        self.index_name = index_name or config.vector_db.index_name
        self.api_key = api_key or config.vector_db.pinecone_api_key
        self.environment = environment or config.vector_db.pinecone_environment
        self.dimension = config.vector_db.dimension
        
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection."""
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Create index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )
            logger.info(f"Created Pinecone index: {self.index_name}")
        
        self.index = pinecone.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors with metadata to Pinecone."""
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = meta.get('chunk_id', f"vec_{i}")
            vectors_to_upsert.append((vector_id, vector.tolist(), meta))
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(vectors)} vectors to Pinecone index")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors in Pinecone."""
        # Query Pinecone
        response = self.index.query(
            vector=query_vector.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        # Prepare results
        results = []
        for match in response['matches']:
            metadata = match['metadata']
            score = match['score']
            results.append((metadata, score))
        
        return results
    
    def save(self, path: str) -> None:
        """Pinecone is cloud-based, no local save needed."""
        logger.info("Pinecone index is automatically persisted in the cloud")
    
    def load(self, path: str) -> None:
        """Pinecone is cloud-based, no local load needed."""
        logger.info("Pinecone index loaded from cloud")


class VectorStoreManager:
    """Manages vector store operations and provides unified interface."""
    
    def __init__(self, store_type: str = None):
        self.store_type = store_type or config.vector_db.type
        self.vector_store = self._create_vector_store()
    
    def _create_vector_store(self) -> VectorStore:
        """Create appropriate vector store based on configuration."""
        if self.store_type.lower() == "faiss":
            return FAISSVectorStore()
        elif self.store_type.lower() == "pinecone":
            return PineconeVectorStore()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def index_documents(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        """Index embedded document chunks."""
        if not embedded_chunks:
            logger.warning("No chunks to index")
            return
        
        # Extract vectors and metadata
        vectors = np.array([chunk['embedding'] for chunk in embedded_chunks])
        metadata = [
            {
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'],
                'document_id': chunk['document_id'],
                'chunk_type': chunk['chunk_type'],
                **chunk['metadata']
            }
            for chunk in embedded_chunks
        ]
        
        # Add to vector store
        self.vector_store.add_vectors(vectors, metadata)
        logger.info(f"Indexed {len(embedded_chunks)} document chunks")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, 
                      filter_criteria: Dict[str, Any] = None, document_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_criteria: Optional filtering criteria
            document_filter: List of document names to filter by
            
        Returns:
            List of similar chunks with scores
        """
        results = self.vector_store.search(query_embedding, k, document_filter)
        
        # Apply additional filtering if specified
        if filter_criteria:
            filtered_results = []
            for metadata, score in results:
                if self._matches_filter(metadata, filter_criteria):
                    filtered_results.append({'metadata': metadata, 'score': score})
            return filtered_results
        
        return [{'metadata': metadata, 'score': score} for metadata, score in results]
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def save_index(self, path: str = None) -> None:
        """Save the vector index."""
        if path is None:
            path = os.path.join(config.embeddings_dir, f"vector_index_{self.store_type}")
        
        self.vector_store.save(path)
    
    def load_index(self, path: str = None) -> None:
        """Load the vector index."""
        if path is None:
            path = os.path.join(config.embeddings_dir, f"vector_index_{self.store_type}")
        
        self.vector_store.load(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if isinstance(self.vector_store, FAISSVectorStore):
            return {
                'type': 'FAISS',
                'total_vectors': self.vector_store.index.ntotal if self.vector_store.index else 0,
                'dimension': self.vector_store.dimension
            }
        elif isinstance(self.vector_store, PineconeVectorStore):
            try:
                stats = self.vector_store.index.describe_index_stats()
                return {
                    'type': 'Pinecone',
                    'total_vectors': stats.get('total_vector_count', 0),
                    'dimension': self.vector_store.dimension
                }
            except Exception as e:
                logger.warning(f"Could not get Pinecone stats: {e}")
                return {'type': 'Pinecone', 'error': str(e)}
        
        return {'type': 'Unknown'}
