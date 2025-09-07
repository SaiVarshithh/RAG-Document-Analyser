"""
Configuration management for the RAG system.
"""
import os
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# api_key: str = Field(default="sk-PNjQOZxluAPEPnkqZtabNg", env="AZURE_AI_API_KEY")
# base_url: str = Field(default="https://genailab.tcs.in", env="AZURE_AI_BASE_URL")
# model: str = Field(default="azure/genailab-maas-gpt-4o", env="AZURE_AI_MODEL")

class GeminiConfig(BaseSettings):
    """Gemini AI configuration."""
    api_key: str = Field(default="", env="GEMINI_API_KEY")
    model: str = Field(default="gemini-2.0-flash", env="GEMINI_MODEL")
    temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    max_tokens: int = Field(default=2000)
    use_fallback: bool = Field(default=True, env="USE_FALLBACK_MODEL")


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""
    type: str = Field(default="faiss", env="VECTOR_DB_TYPE")
    index_name: str = Field(default="technical-docs")
    dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=32)


class ChunkingConfig(BaseSettings):
    """Text chunking configuration."""
    max_chunk_size: int = Field(default=1500, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(default=300, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=50, env="MIN_CHUNK_SIZE")
    separators: list = Field(default=["\n\n", "\n", ".", "!", "?", ";", " ", ""])


class APIConfig(BaseSettings):
    """API server configuration."""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    cors_origins: list = Field(default=["*"])


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    file: str = Field(default="logs/rag_system.log", env="LOG_FILE")
    format: str = Field(default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.gemini = GeminiConfig()
        self.vector_db = VectorDBConfig()
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        
        # Data paths
        self.data_dir = "data"
        self.documents_dir = os.path.join(self.data_dir, "documents")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.embeddings_dir = os.path.join(self.data_dir, "embeddings")
        
        # Ensure directories exist
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)


# Global configuration instance
config = Config()
