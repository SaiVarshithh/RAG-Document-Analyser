"""
Main RAG system application logic.
"""
import os
from typing import List, Dict, Any
from loguru import logger

from config import config
from src.ingestion.document_processor import DocumentProcessor
from src.chunking.text_chunker import SmartTextChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStoreManager
from src.generation.llm_generator import LLMGenerator


class RAGSystem:
    """Orchestrates the entire RAG pipeline."""

    def __init__(self):
        logger.add(config.logging.file, level=config.logging.level, format=config.logging.format)
        logger.info("Initializing RAG System...")

        self.doc_processor = DocumentProcessor()
        self.chunker = SmartTextChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_store_manager = VectorStoreManager()
        self.llm_generator = LLMGenerator()
        self.is_index_loaded = False

    def setup_pipeline(self, documents_path: str):
        """
        Run the full ingestion and indexing pipeline.

        Args:
            documents_path: Path to the directory containing documents.
        """
        logger.info(f"Starting ingestion from directory: {documents_path}")
        
        # 1. Process documents
        documents = self.doc_processor.process_directory(documents_path)
        if not documents:
            logger.warning("No documents were processed. Aborting pipeline.")
            return

        # 2. Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks were created from the documents. Aborting pipeline.")
            return

        # 3. Generate embeddings
        embedded_chunks = self.embedder.generate_embeddings(all_chunks)

        # 4. Index in vector store
        self.vector_store_manager.index_documents(embedded_chunks)

        # 5. Save the index
        self.vector_store_manager.save_index()
        self.is_index_loaded = True
        logger.info("Pipeline setup complete. Index is ready.")

    def ask_question(self, query: str, top_k: int = 15, selected_documents: List[str] = None, chat_history: str = "") -> Dict[str, Any]:
        """
        Ask a question and get an answer from the RAG system.
        
        Args:
            query: The user's question
            top_k: Number of relevant chunks to retrieve
            selected_documents: List of document names to filter search (optional)
            chat_history: Previous conversation context for memory (optional)
        
        Returns:
            Dictionary containing the answer and source information
        """
        # Validate query input
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Received empty or invalid query")
            return {
                "answer": "Please provide a valid question to search through your documents.",
                "sources": []
            }
        
        # Clean the query
        query = query.strip()
        
        
        if not self.vector_store_manager.vector_store.index:
            try:
                self.vector_store_manager.load_index()
                logger.info("Index loaded successfully.")
            except FileNotFoundError:
                logger.error("Index not found. Please run the setup_pipeline first.")
                return {
                    "answer": "The document index has not been created yet. Please ingest documents first.",
                    "sources": []
                }

        logger.info(f"Received query: {query}")

        # 1. Generate query embedding
        query_embedding = self.embedder.generate_query_embedding(query)

        # 2. Retrieve relevant chunks with maximum coverage for summarization and analysis
        retrieved_chunks = self.vector_store_manager.search_similar(
            query_embedding, k=top_k * 3, document_filter=selected_documents
        )
        
        # For summarization requests, get even more chunks
        if any(word in query.lower() for word in ['summarize', 'summary', 'overview', 'list', 'all']):
            logger.info("Detected summarization/listing request, retrieving more chunks")
            retrieved_chunks = self.vector_store_manager.search_similar(
                query_embedding, k=top_k * 5, document_filter=selected_documents
            )
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

        # 3. Generate response with conversational memory - always call LLM even with empty chunks
        response = self.llm_generator.generate_response(
            query, retrieved_chunks, chat_history=chat_history, selected_documents=selected_documents
        )

        logger.info(f"Generated answer for query: '{query}'")
        return response
    
