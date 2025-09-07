# RAG Document Analyzer - User Documentation

## 📚 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Installation & Setup](#installation--setup)
5. [User Guide](#user-guide)
6. [Technical Details](#technical-details)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## 🎯 Overview

The **RAG Document Analyzer** is an advanced AI-powered system that enables users to upload technical documents and ask natural language questions about their content. Using Retrieval-Augmented Generation (RAG) technology, the system provides accurate, context-aware answers with source citations.

### Key Benefits
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, HTML
- **Conversational Memory**: Maintains context across chat sessions
- **Document Selection**: Filter searches to specific documents
- **Source Attribution**: Shows exactly where answers come from
- **Real-time Processing**: Fast semantic search and response generation

---

## ✨ Features

### 🔍 Document Processing
- **Multi-format Ingestion**: Supports PDF, DOCX, TXT, MD, HTML files
- **Intelligent Chunking**: Breaks documents into semantic chunks for better retrieval
- **Vector Embeddings**: Uses sentence-transformers for high-quality embeddings
- **FAISS Indexing**: Fast similarity search with FAISS vector database

### 💬 Conversational AI
- **Natural Language Queries**: Ask questions in plain English
- **Conversational Memory**: Remembers previous questions and context
- **Document-focused Responses**: Only answers based on uploaded documents
- **Source Citations**: Shows relevant document sections for each answer

### 🎛️ User Interface
- **Streamlit Web Interface**: Clean, intuitive web application
- **Document Selection**: Choose which documents to search
- **Chat History**: View and manage conversation history
- **New Chat Feature**: Start fresh conversations easily

### 🔧 Advanced Features
- **Dynamic Document Filtering**: Search specific documents or all at once
- **Context-aware Responses**: Uses chat history for better understanding
- **Error Handling**: Graceful handling of edge cases and errors
- **Configurable Models**: Easy to swap LLM and embedding models

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │  Document       │    │   Response      │
│   (Questions)   │───▶│  Processing     │───▶│   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        ▲
                              ▼                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector        │    │      LLM        │
│   Upload        │───▶│   Database      │───▶│   (Azure AI)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

```
RAG Document Analyzer
├── Frontend (Streamlit)
│   ├── Document Upload Interface
│   ├── Chat Interface
│   ├── Document Selection
│   └── New Chat Management
│
├── Backend Processing
│   ├── Document Ingestion
│   │   ├── PDF Processor
│   │   ├── DOCX Processor
│   │   ├── Text Processor
│   │   └── HTML/Markdown Processor
│   │
│   ├── Text Processing
│   │   ├── Text Chunking
│   │   ├── Metadata Extraction
│   │   └── Content Cleaning
│   │
│   ├── Vector Processing
│   │   ├── Embedding Generation
│   │   ├── FAISS Indexing
│   │   └── Similarity Search
│   │
│   └── Response Generation
│       ├── Context Retrieval
│       ├── Prompt Engineering
│       ├── LLM Integration
│       └── Source Attribution
│
└── Configuration & Storage
    ├── Environment Variables
    ├── Model Configuration
    ├── Vector Index Storage
    └── Logging System
```

### Data Flow Diagram

```
1. Document Upload
   ┌─────────────┐
   │   User      │
   │  Uploads    │──┐
   │ Documents   │  │
   └─────────────┘  │
                    ▼
   ┌─────────────────────────────┐
   │    Document Processor       │
   │  ┌─────────┬─────────────┐  │
   │  │   PDF   │    DOCX     │  │
   │  │ Parser  │   Parser    │  │
   │  └─────────┴─────────────┘  │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │      Text Chunker           │
   │   ┌─────────────────────┐   │
   │   │  Semantic Chunks    │   │
   │   │  + Metadata         │   │
   │   └─────────────────────┘   │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │   Embedding Generator       │
   │  ┌─────────────────────┐    │
   │  │ sentence-transformers│    │
   │  │   (384 dimensions)  │    │
   │  └─────────────────────┘    │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │     FAISS Vector Store      │
   │  ┌─────────────────────┐    │
   │  │   Vector Index      │    │
   │  │   + Metadata        │    │
   │  └─────────────────────┘    │
   └─────────────────────────────┘

2. Query Processing
   ┌─────────────┐
   │    User     │
   │   Query     │──┐
   └─────────────┘  │
                    ▼
   ┌─────────────────────────────┐
   │    Query Embedding          │
   │  ┌─────────────────────┐    │
   │  │  Same Embedding     │    │
   │  │     Model           │    │
   │  └─────────────────────┘    │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │   Similarity Search         │
   │  ┌─────────────────────┐    │
   │  │  FAISS Search       │    │
   │  │  Top-K Results      │    │
   │  └─────────────────────┘    │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │   Context Assembly          │
   │  ┌─────────────────────┐    │
   │  │  Retrieved Chunks   │    │
   │  │  + Chat History     │    │
   │  │  + Selected Docs    │    │
   │  └─────────────────────┘    │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │      LLM Generation         │
   │  ┌─────────────────────┐    │
   │  │   Azure AI LLM      │    │
   │  │  (GPT-4o/DeepSeek)  │    │
   │  └─────────────────────┘    │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐
   │    Response + Sources       │
   │  ┌─────────────────────┐    │
   │  │   Final Answer      │    │
   │  │   + Citations       │    │
   │  └─────────────────────┘    │
   └─────────────────────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Azure AI API access (for LLM)
- 4GB+ RAM (for embedding models)
- 2GB+ disk space

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd rag-document-analyzer
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration
Create a `.env` file in the project root:

```env
# Azure AI Configuration
AZURE_AI_API_KEY=your_azure_api_key_here
AZURE_AI_BASE_URL=https://your-endpoint.openai.azure.com/
AZURE_AI_MODEL=azure/genailab-maas-gpt-4o

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Vector Database Configuration
VECTOR_DB_TYPE=faiss

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS_PER_DOCUMENT=100

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Step 4: Run the Application
```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

---

## 📖 User Guide

### Getting Started

#### 1. Upload Documents
1. Open the application in your web browser
2. Use the sidebar file uploader to select documents
3. Supported formats: PDF, DOCX, TXT, MD, HTML
4. Multiple files can be uploaded simultaneously

#### 2. Process Documents
1. Click "Process Documents" button in the sidebar
2. Wait for processing to complete (may take a few minutes)
3. Success message will confirm when ready

#### 3. Select Documents (Optional)
1. Use the multi-select dropdown to choose specific documents
2. Leave empty to search all uploaded documents
3. Selection can be changed anytime during conversation

#### 4. Start Asking Questions
1. Type your question in the chat input
2. Press Enter to submit
3. View the AI response with source citations
4. Continue the conversation naturally

#### 5. Manage Conversations
1. Use "New Chat" button to start fresh conversations
2. Chat history is maintained within each session
3. Document selection is preserved across new chats

### Advanced Features

#### Document Selection
- **All Documents**: Leave selection empty to search everything
- **Specific Documents**: Select particular files for focused search
- **Dynamic Switching**: Change selection mid-conversation
- **Context Awareness**: System tracks selection changes

#### Conversational Memory
- **Context Retention**: Remembers previous questions and answers
- **Follow-up Questions**: Understands references to earlier topics
- **Session Persistence**: Memory maintained until "New Chat"
- **Smart Prompting**: Uses history for better responses

#### Source Attribution
- **Relevance Scores**: Shows how well sources match your question
- **Content Preview**: Displays snippet from source document
- **File Names**: Identifies which document contains the information
- **Expandable Sources**: Click to see detailed source information

### Best Practices

#### Document Preparation
- **Clear Text**: Ensure documents have readable text (not just images)
- **Structured Content**: Well-organized documents work better
- **Reasonable Size**: Very large documents may take longer to process
- **Multiple Formats**: Mix different file types as needed

#### Question Formulation
- **Be Specific**: Clear, specific questions get better answers
- **Use Context**: Reference previous questions when appropriate
- **Ask Follow-ups**: Build on previous answers naturally
- **Check Sources**: Review citations to verify information

#### System Usage
- **Process Once**: Documents only need processing once per session
- **Select Wisely**: Use document selection for focused searches
- **New Chat**: Start fresh when switching topics completely
- **Monitor Performance**: Large document sets may be slower

---

## 🔧 Technical Details

### Supported Models

#### Large Language Models (LLMs)
- **Azure AI GPT-4o**: Default recommended model
- **DeepSeek-V3**: Alternative high-performance model
- **Custom Models**: Configurable via environment variables

#### Embedding Models
- **sentence-transformers/all-MiniLM-L6-v2**: Default (384 dimensions)
- **Alternative Models**: Any sentence-transformers compatible model
- **Custom Embeddings**: Configurable dimension and model

### Vector Database
- **FAISS**: Facebook AI Similarity Search (default)
- **Local Storage**: Indexes stored locally for fast access
- **Scalable**: Handles thousands of documents efficiently

### Document Processing Pipeline

#### 1. Ingestion
```python
# Supported file types and processors
processors = {
    'pdf': PyPDF2 + pdfplumber,
    'docx': python-docx,
    'txt': Built-in text reader,
    'md': Markdown parser,
    'html': BeautifulSoup parser
}
```

#### 2. Text Chunking
```python
# Chunking strategy
chunk_size = 1000  # characters
chunk_overlap = 200  # characters
max_chunks = 100  # per document
```

#### 3. Embedding Generation
```python
# Embedding process
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)  # 384-dimensional vectors
```

#### 4. Vector Indexing
```python
# FAISS index creation
index = faiss.IndexFlatIP(384)  # Inner product similarity
index.add(embeddings)  # Add all vectors
```

### Response Generation Pipeline

#### 1. Query Processing
```python
# Query embedding
query_embedding = embedding_model.encode(query)
```

#### 2. Similarity Search
```python
# Retrieve top-k similar chunks
scores, indices = index.search(query_embedding, k=5)
relevant_chunks = [chunks[i] for i in indices[0]]
```

#### 3. Context Assembly
```python
# Build context for LLM
context = {
    'retrieved_chunks': relevant_chunks,
    'chat_history': previous_conversation,
    'selected_documents': user_selection,
    'query': user_question
}
```

#### 4. LLM Generation
```python
# Generate response using LLM
prompt = create_prompt_template(context)
response = llm.invoke(prompt)
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AZURE_AI_API_KEY` | Azure AI API key | - | Yes |
| `AZURE_AI_BASE_URL` | Azure AI endpoint URL | - | Yes |
| `AZURE_AI_MODEL` | LLM model name | `azure/genailab-maas-gpt-4o` | No |
| `EMBEDDING_MODEL` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` | No |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `384` | No |
| `VECTOR_DB_TYPE` | Vector database type | `faiss` | No |
| `CHUNK_SIZE` | Text chunk size | `1000` | No |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` | No |
| `MAX_CHUNKS_PER_DOCUMENT` | Max chunks per document | `100` | No |

### Model Configuration

#### Changing LLM Model
```env
# Use different Azure AI model
AZURE_AI_MODEL=azure/genailab-maas-DeepSeek-V3-0324
```

#### Changing Embedding Model
```env
# Use different embedding model
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

### Performance Tuning

#### For Large Documents
```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
MAX_CHUNKS_PER_DOCUMENT=200
```

#### For Better Accuracy
```env
CHUNK_SIZE=800
CHUNK_OVERLAP=150
# Smaller chunks for more precise retrieval
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. API Key Errors
**Problem**: `Authentication failed` or `Invalid API key`
**Solution**: 
- Verify Azure AI API key is correct
- Check API endpoint URL format
- Ensure API key has proper permissions

#### 2. Document Processing Fails
**Problem**: `Error processing documents`
**Solutions**:
- Check file format is supported (PDF, DOCX, TXT, MD, HTML)
- Ensure files are not corrupted
- Try processing smaller batches
- Check available disk space

#### 3. Slow Response Times
**Problem**: Queries take too long to respond
**Solutions**:
- Reduce number of documents
- Use document selection to filter search
- Check internet connection for API calls
- Consider upgrading hardware (RAM/CPU)

#### 4. Empty or Poor Responses
**Problem**: AI gives irrelevant or "I don't know" responses
**Solutions**:
- Ensure documents contain relevant information
- Try rephrasing questions more specifically
- Check document processing completed successfully
- Verify document selection includes relevant files

#### 5. Memory Issues
**Problem**: `Out of memory` errors during processing
**Solutions**:
- Process fewer documents at once
- Reduce `MAX_CHUNKS_PER_DOCUMENT` setting
- Close other applications to free RAM
- Use smaller embedding models if available

### Debug Mode

Enable detailed logging by setting:
```env
LOG_LEVEL=DEBUG
```

This will provide detailed information about:
- Document processing steps
- Embedding generation progress
- Vector search results
- LLM API calls and responses

### Performance Monitoring

Check system performance using built-in metrics:
- Document processing time
- Embedding generation speed
- Vector search latency
- LLM response time

---

## 📚 API Reference

### Core Classes

#### RAGSystem
Main orchestrator class that coordinates all components.

```python
class RAGSystem:
    def __init__(self):
        """Initialize RAG system with all components."""
        
    def setup_pipeline(self, documents_dir: str):
        """Process documents and create vector index."""
        
    def ask_question(self, query: str, selected_documents: List[str] = None, 
                    chat_history: str = "") -> Dict[str, Any]:
        """Ask a question and get response with sources."""
```

#### DocumentProcessor
Handles document ingestion and text extraction.

```python
class DocumentProcessor:
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process single document and extract text."""
        
    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Process all documents in directory."""
```

#### EmbeddingGenerator
Generates vector embeddings for text chunks.

```python
class EmbeddingGenerator:
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for document chunks."""
        
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for user query."""
```

#### VectorStoreManager
Manages vector database operations.

```python
class VectorStoreManager:
    def index_documents(self, embedded_chunks: List[Dict]):
        """Index document chunks in vector database."""
        
    def search_similar(self, query_embedding: np.ndarray, k: int = 5,
                      document_filter: List[str] = None) -> List[Dict]:
        """Search for similar chunks."""
```

#### LLMGenerator
Handles LLM integration and response generation.

```python
class LLMGenerator:
    def generate_response(self, context: str, query: str, 
                         chat_history: str = "", 
                         selected_documents: str = "") -> str:
        """Generate response using LLM."""
```

### Configuration Classes

#### Config
Central configuration management.

```python
class Config:
    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_document: int = 100
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Azure AI settings
    azure_ai_api_key: str
    azure_ai_base_url: str
    azure_ai_model: str = "azure/genailab-maas-gpt-4o"
```

### Response Format

#### Question Response
```python
{
    "answer": "Generated response text",
    "sources": [
        {
            "file_name": "document.pdf",
            "content": "Relevant text chunk",
            "score": 0.85,
            "chunk_id": "doc_001_chunk_005",
            "metadata": {
                "document_id": "document.pdf",
                "chunk_type": "paragraph"
            }
        }
    ]
}
```

---

## 🤝 Support & Contributing

### Getting Help
- Check this documentation first
- Review troubleshooting section
- Check GitHub issues for similar problems
- Create new issue with detailed description

### Contributing
- Fork the repository
- Create feature branch
- Make changes with tests
- Submit pull request with description

### License
This project is licensed under the MIT License. See LICENSE file for details.

---

## 📝 Changelog

### Version 1.0.0
- Initial release with core RAG functionality
- Multi-format document support
- Conversational memory
- Document selection filtering
- Streamlit web interface
- Azure AI LLM integration
- FAISS vector database
- Source attribution

---

*Last updated: January 2025*
