# 🚀 Quick Start Guide - RAG Document Analyzer

## ⚡ Get Started in 5 Minutes

### 1. Prerequisites
- Python 3.8+
- Azure AI API access
- 4GB+ RAM

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd rag-document-analyzer

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create `.env` file:
```env
AZURE_AI_API_KEY=your_api_key_here
AZURE_AI_BASE_URL=https://your-endpoint.openai.azure.com/
AZURE_AI_MODEL=azure/genailab-maas-gpt-4o
```

### 4. Run Application
```bash
streamlit run streamlit_app.py
```

### 5. Use the System
1. **Upload Documents** - PDF, DOCX, TXT, MD, HTML files
2. **Process Documents** - Click "Process Documents" button
3. **Ask Questions** - Type natural language questions
4. **Get Answers** - Receive AI responses with source citations

## 🎯 Example Usage

### Sample Questions
- "What are the main features described in the documentation?"
- "How do I configure the authentication system?"
- "What are the system requirements?"
- "Summarize the installation process"

### Advanced Features
- **Document Selection**: Choose specific documents to search
- **Conversational Memory**: Ask follow-up questions naturally
- **New Chat**: Start fresh conversations anytime
- **Source Citations**: See exactly where answers come from

## 📚 Next Steps
- Read the [User Documentation](USER_DOCUMENTATION.md) for detailed usage
- Check the [Architecture Guide](ARCHITECTURE_GUIDE.md) for technical details
- Review configuration options for optimization

## 🆘 Need Help?
- Check the troubleshooting section in the user documentation
- Verify your Azure AI API credentials
- Ensure documents are in supported formats
- Monitor system resources during processing

---
*Ready to analyze your documents with AI? Let's get started!* 🚀
