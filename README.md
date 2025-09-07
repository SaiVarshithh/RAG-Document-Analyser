# RAG Document Analyzer

A comprehensive Retrieval-Augmented Generation (RAG) system for technical documentation analysis and question-answering. This system allows users to upload documents and ask natural language questions, receiving accurate answers with source citations.

## Features

- **Multi-format Document Support**: PDF, HTML, Markdown, DOCX, and TXT files
- **Semantic Search**: Vector-based search using FAISS for fast, accurate retrieval
- **Strict Document-Based Responses**: AI answers only from uploaded documents, no external knowledge
- **Source Attribution**: Every answer includes citations to source documents
- **Interactive Web Interface**: User-friendly Streamlit interface with chat functionality
- **Real-time Processing**: Upload and process documents instantly
- **Relevance Scoring**: See how relevant each source is to your question

## 🏗️ Architecture

### Core Components

1. **Document Ingestion** (`src/ingestion/`): Multi-format text extraction
2. **Text Chunking** (`src/chunking/`): Intelligent document segmentation with overlap
3. **Embedding Generation** (`src/embeddings/`): Vector representations using sentence-transformers
4. **Vector Storage** (`src/retrieval/`): FAISS-based semantic search index
5. **Response Generation** (`src/generation/`): LLM-powered answer generation with strict document adherence
6. **Web Interface** (`streamlit_app.py`): Interactive chat interface

### Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: FAISS (local, no external dependencies)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Google Gemini (with fallback support)
- **Document Processing**: PyPDF2, pdfplumber, BeautifulSoup, python-docx
- **Framework**: LangChain for LLM orchestration

## 📦 Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Google Gemini API key (optional, works without for demo)

### Setup

1. **Clone and navigate to the project**:
```bash
cd RAG-Document-Analyser
```

2. **Create virtual environment**:
```bash
python -m venv rag_venv
rag_venv\Scripts\activate  # On Windows
# source rag_venv/bin/activate  # On Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env with your Gemini API key for full functionality
```

## Usage

### Web Interface (Recommended)

1. **Start the application**:
```bash
streamlit run streamlit_app.py
```

2. **Access the interface**:
   - Open your browser to `http://localhost:8501`
   - Upload your documents using the sidebar
   - Click "Process Documents" to index them
   - Start asking questions in the chat interface

### Command Line Interface

1. **Process documents**:
```bash
python src/main.py ingest --path data/documents
```

2. **Ask questions**:
```bash
python src/main.py ask "What are the main features described in the documentation?"
```

## 📋 Configuration

The system uses the following key configurations:

### Document Processing (Optimized)
- **Chunk Size**: 1500 characters (optimized for better retrieval)
- **Chunk Overlap**: 300 characters for enhanced context continuity
- **Supported Formats**: PDF, DOCX, TXT, MD, HTML

### AI Models
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: Google Gemini 2.0 Flash (with fallback support)
- **Vector Database**: FAISS with cosine similarity
- **Retrieval**: Enhanced with 30+ chunks for comprehensive coverage

### Environment Variables (Optional)
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.1
USE_FALLBACK_MODEL=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 🔍 How It Works

### Document Processing Flow
1. **Upload**: Documents uploaded via web interface
2. **Extraction**: Text extracted based on file type
3. **Chunking**: Documents split into overlapping chunks
4. **Embedding**: Each chunk converted to 384-dimensional vectors
5. **Indexing**: Vectors stored in FAISS index for fast search

### Query Processing Flow
1. **Question**: User asks a natural language question
2. **Embedding**: Question converted to vector representation
3. **Search**: Top-k most similar document chunks retrieved
4. **Generation**: LLM generates answer using only retrieved context
5. **Response**: Answer displayed with source citations

## Key Features

### Strict Document Adherence
The system is configured to:
- Only use information from uploaded documents
- Refuse to answer questions not covered in documents
- Provide clear "I don't know" responses when information is unavailable
- Always cite source documents for transparency

### Example Interactions
```
User: "What are the system requirements?"
AI: "Based on the documentation, the system requirements are... [Source: requirements.pdf]"

User: "What's the weather like today?"
AI: "I don't know. This question is not related to the content in the provided documents."
```

## Project Structure

```
rag-document-analyzer/
├── streamlit_app.py          # Main web interface
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env.example             # Environment configuration template
├── src/
│   ├── app.py              # Core RAG system orchestrator
│   ├── main.py             # CLI interface
│   ├── config.py           # Configuration management
│   ├── ingestion/          # Document processing
│   ├── chunking/           # Text segmentation
│   ├── embeddings/         # Vector generation
│   ├── retrieval/          # Search and indexing
│   └── generation/         # LLM response generation
├── data/
│   ├── documents/          # Uploaded documents
│   ├── processed/          # Processed document cache
│   └── embeddings/         # Vector index storage
└── logs/                   # Application logs
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Memory Issues**: For large documents, consider reducing chunk size in configuration
3. **Slow Processing**: First-time model download may take several minutes
4. **Port Conflicts**: Change Streamlit port with `--server.port 8502`

### Performance Tips

- **GPU Acceleration**: Install PyTorch with CUDA support for faster embeddings
- **Batch Processing**: Use CLI for processing large document collections
- **Memory Management**: Process documents in smaller batches if memory is limited

## Support

For questions or issues, please:
1. Review the logs in the `logs/` directory
