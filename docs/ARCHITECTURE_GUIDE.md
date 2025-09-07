# RAG Document Analyzer - Architecture Guide

## 🏗️ System Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Document Analyzer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Frontend   │    │   Backend   │    │  External   │         │
│  │ (Streamlit) │◄──►│ Processing  │◄──►│  Services   │         │
│  │             │    │             │    │ (Azure AI)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Document    │    │   Vector    │    │   Config    │         │
│  │ Storage     │    │  Database   │    │  Storage    │         │
│  │ (Local)     │    │  (FAISS)    │    │ (.env)      │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Component Architecture

### Core Components Breakdown

```
RAG System Components
│
├── 🖥️  User Interface Layer
│   ├── Streamlit Web App
│   ├── Document Upload Interface
│   ├── Chat Interface
│   └── Configuration Panel
│
├── 🔄 Processing Layer
│   ├── Document Processor
│   │   ├── PDF Handler
│   │   ├── DOCX Handler
│   │   ├── Text Handler
│   │   └── HTML/MD Handler
│   │
│   ├── Text Chunker
│   │   ├── Semantic Splitting
│   │   ├── Overlap Management
│   │   └── Metadata Extraction
│   │
│   ├── Embedding Generator
│   │   ├── SentenceTransformers
│   │   ├── Vector Generation
│   │   └── Batch Processing
│   │
│   └── Response Generator
│       ├── Context Assembly
│       ├── Prompt Engineering
│       ├── LLM Integration
│       └── Source Attribution
│
├── 💾 Storage Layer
│   ├── Vector Store (FAISS)
│   ├── Document Storage
│   ├── Index Management
│   └── Metadata Storage
│
└── 🔧 Configuration Layer
    ├── Environment Variables
    ├── Model Configuration
    ├── API Settings
    └── Performance Tuning
```

## 🔄 Data Flow Architecture

### Document Processing Flow

```
📄 Document Upload
        │
        ▼
┌───────────────────┐
│   File Validator  │ ──► Checks format, size, type
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Format Processor  │ ──► PDF/DOCX/TXT/MD/HTML parsers
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Text Extractor   │ ──► Clean text extraction
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Text Chunker    │ ──► Semantic chunking + metadata
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Embedding Engine  │ ──► sentence-transformers encoding
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Vector Storage   │ ──► FAISS index creation
└───────────────────┘
```

### Query Processing Flow

```
❓ User Question
        │
        ▼
┌───────────────────┐
│ Query Validation  │ ──► Check empty, format, length
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Query Embedding   │ ──► Same embedding model as docs
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Similarity Search │ ──► FAISS vector search (top-k)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Context Assembly  │ ──► Combine chunks + history + filters
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Prompt Creation  │ ──► Build LLM prompt with context
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ LLM Generation    │ ──► Azure AI API call
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Response Format   │ ──► Answer + sources + metadata
└───────────────────┘
```

## 🧩 Module Dependencies

### Dependency Graph

```
streamlit_app.py (Entry Point)
        │
        ▼
    app.py (RAG System)
        │
        ├─► document_processor.py
        │   └─► PyPDF2, python-docx, BeautifulSoup
        │
        ├─► text_chunker.py
        │   └─► langchain.text_splitter
        │
        ├─► embedding_generator.py
        │   └─► sentence-transformers
        │
        ├─► vector_store.py
        │   └─► faiss-cpu, numpy
        │
        ├─► llm_generator.py
        │   └─► langchain, openai
        │
        └─► config.py
            └─► python-dotenv
```

## 🔐 Security Architecture

### Security Layers

```
┌─────────────────────────────────────────┐
│            Security Layers              │
├─────────────────────────────────────────┤
│ 🔒 API Key Management                   │
│    ├── Environment Variables           │
│    ├── No Hardcoded Secrets           │
│    └── Azure AI Authentication        │
├─────────────────────────────────────────┤
│ 📁 File Security                       │
│    ├── Type Validation                │
│    ├── Size Limits                    │
│    └── Local Storage Only             │
├─────────────────────────────────────────┤
│ 🛡️ Input Validation                    │
│    ├── Query Sanitization             │
│    ├── Empty Input Handling           │
│    └── Error Boundaries               │
├─────────────────────────────────────────┤
│ 🔍 Data Privacy                        │
│    ├── Local Processing               │
│    ├── No Data Persistence            │
│    └── Session-based Storage          │
└─────────────────────────────────────────┘
```

## ⚡ Performance Architecture

### Performance Optimization Strategies

```
🚀 Performance Layers
│
├── 📊 Caching Strategy
│   ├── Vector Index Caching
│   ├── Embedding Model Caching
│   └── Session State Management
│
├── 🔄 Batch Processing
│   ├── Document Batch Upload
│   ├── Embedding Batch Generation
│   └── Parallel Processing
│
├── 🎯 Search Optimization
│   ├── FAISS Index Optimization
│   ├── Top-K Result Limiting
│   └── Document Filtering
│
└── 💾 Memory Management
    ├── Streaming Document Processing
    ├── Chunk Size Optimization
    └── Garbage Collection
```

## 🔧 Deployment Architecture

### Deployment Options

```
┌─────────────────────────────────────────┐
│          Deployment Strategies          │
├─────────────────────────────────────────┤
│ 🖥️ Local Development                    │
│    ├── Streamlit run                   │
│    ├── Local file storage             │
│    └── Development server             │
├─────────────────────────────────────────┤
│ ☁️ Cloud Deployment                     │
│    ├── Streamlit Cloud                │
│    ├── Heroku                         │
│    ├── AWS/Azure/GCP                  │
│    └── Docker Containers              │
├─────────────────────────────────────────┤
│ 🏢 Enterprise Deployment               │
│    ├── Kubernetes                     │
│    ├── Load Balancing                 │
│    ├── Auto Scaling                   │
│    └── High Availability              │
└─────────────────────────────────────────┘
```

## 📈 Scalability Considerations

### Horizontal Scaling

```
Single Instance → Multiple Instances
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   RAG App   │    │   RAG App   │    │   RAG App   │
│  Instance 1 │    │  Instance 2 │    │  Instance N │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌─────────────┐
                    │ Load Balancer│
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │ Shared Vector│
                    │   Database   │
                    └─────────────┘
```

### Vertical Scaling

```
Resource Scaling
├── 💾 Memory Scaling
│   ├── Larger embedding models
│   ├── More document capacity
│   └── Bigger vector indices
│
├── 🖥️ CPU Scaling
│   ├── Faster document processing
│   ├── Parallel embedding generation
│   └── Concurrent user handling
│
└── 💿 Storage Scaling
    ├── More document storage
    ├── Larger vector indices
    └── Extended chat history
```

This architecture guide provides the technical foundation for understanding how the RAG Document Analyzer system works, scales, and can be deployed in various environments.
