"""
Response generation using the specified LLM.
"""
from typing import List, Dict, Any
import os
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from loguru import logger

from config import config


class LLMGenerator:
    """Handles LLM-based response generation."""

    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()

    def _initialize_llm(self):
        """Initialize the Gemini client with fallback options."""
        try:
            # Pull live values from environment (set by Streamlit UI) with config defaults as fallback
            env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
            api_key = env_api_key if env_api_key else (config.gemini.api_key or "")
            env_temp = os.getenv("GEMINI_TEMPERATURE")
            try:
                temperature = float(env_temp) if env_temp is not None else float(config.gemini.temperature)
            except Exception:
                temperature = float(config.gemini.temperature)

            logger.info(f"Initializing Gemini LLM: {config.gemini.model}")
            
            # Check if API key is provided
            if not api_key:
                logger.warning("No Gemini API key provided. Using mock responses for demonstration.")
                return self._create_mock_llm()
            
            llm = ChatGoogleGenerativeAI(
                model=config.gemini.model,
                google_api_key=api_key,
                temperature=temperature,
                max_tokens=config.gemini.max_tokens,
                convert_system_message_to_human=True
            )
            
            # Test the connection with a simple call
            try:
                _ = llm.invoke("Hello")
                logger.info("Gemini LLM initialized and tested successfully.")
                return llm
            except Exception as test_error:
                logger.warning(f"Gemini LLM test failed: {test_error}")
                if config.gemini.use_fallback:
                    logger.info("Falling back to mock responses.")
                    return self._create_mock_llm()
                else:
                    raise test_error
                    
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            if config.gemini.use_fallback:
                logger.info("Using fallback mock responses.")
                return self._create_mock_llm()
            else:
                raise

    def _create_mock_llm(self):
        """Create a mock LLM for fallback using RunnableLambda."""
        def mock_fn(prompt: str) -> str:
            return (
                "Demo mode: AI service not configured.\n\n"
                "I can still help you navigate your documents. Try providing more specific questions.\n\n"
                "To enable full answers, add your Gemini API key in the left sidebar."
            )
        return RunnableLambda(lambda x: mock_fn(x))

    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the RAG chain with conversational memory."""
        template = """
        You are a document analysis assistant. You must ONLY answer based on the provided context below.

        CRITICAL RULES:
        1. ONLY use information from the CONTEXT provided below
        2. DO NOT use any external knowledge, training data, or general information
        3. If the CONTEXT contains relevant information, answer directly from it WITHOUT any recommendations
        4. If the CONTEXT does not contain ANY relevant information to answer the question:
            - If the input appears to be casual conversation (greetings, acknowledgments, small talk), respond naturally and conversationally
            - If the input is a genuine question but no relevant information is found, respond:
                "I'm sorry, but I could not find any relevant information about that in the provided documents."
                Then suggest 2-3 relevant questions based on the available content
        5. DO NOT provide any information that is not explicitly stated in the CONTEXT
        6. DO NOT add general knowledge or explanations beyond what is in the CONTEXT
        7. DO NOT give recommendations when you CAN answer from the context
        8. Use the CHAT HISTORY to understand context and follow-up questions, but still only answer from the CONTEXT

        SELECTED DOCUMENTS:
        {selected_documents}

        CHAT HISTORY:
        {chat_history}

        CONTEXT:
        {context}

        CURRENT QUESTION:
        {question}

        ANSWER (based STRICTLY on the context above, considering chat history for better understanding):
        """
        return PromptTemplate(template=template, input_variables=["context", "question", "chat_history", "selected_documents"])

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format the retrieved chunks into a string for the prompt."""
        if not retrieved_chunks:
            return "No specific document context available."
        
        formatted_context = ""
        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown')
            content = metadata.get('content', '')
            formatted_context += f"Source [{i+1}]: {source_file}\n"
            formatted_context += f"Content: {content}\n\n"
        return formatted_context

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                         chat_history: str = "", selected_documents: List[str] = None) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved context with conversational memory.

        Args:
            query: The user's question.
            retrieved_chunks: A list of relevant document chunks from the vector store.
            chat_history: Previous conversation context.
            selected_documents: List of currently selected document names.

        Returns:
            A dictionary containing the generated answer and the sources.
        """

        formatted_context = self._format_context(retrieved_chunks)
        
        # Format selected documents
        selected_docs_str = ", ".join(selected_documents) if selected_documents else "All documents"
        
        # Create the RAG chain using LangChain Expression Language (LCEL) with conversational memory
        rag_chain = (
            {
                "context": (lambda x: formatted_context), 
                "question": RunnablePassthrough(),
                "chat_history": (lambda x: chat_history),
                "selected_documents": (lambda x: selected_docs_str)
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        logger.info("Invoking RAG chain to generate the final answer with conversational memory.")
        answer = rag_chain.invoke(query)
        # Safety: extract content from various possible return types
        try:
            if hasattr(answer, "content"):
                answer = answer.content
            elif isinstance(answer, (list, tuple)) and answer and hasattr(answer[0], "content"):
                answer = answer[0].content
            elif isinstance(answer, dict) and "content" in answer:
                answer = str(answer.get("content", ""))
            elif not isinstance(answer, str):
                answer = str(answer)
        except Exception:
            answer = str(answer)

        # Prepare source attribution
        sources = [
            {
                "file_name": chunk.get('metadata', {}).get('source_file', 'Unknown'),
                "content": chunk.get('metadata', {}).get('content', ''),
                "score": chunk.get('score', 0.0)
            }
            for chunk in retrieved_chunks
        ]

        return {"answer": answer, "sources": sources}
