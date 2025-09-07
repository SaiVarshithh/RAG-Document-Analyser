"""
Streamlit web interface for the RAG system.
"""
import os
import streamlit as st
from pathlib import Path
import sys

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Now import the RAG components
from app import RAGSystem

# Set page configuration
st.set_page_config(
    page_title="RAG Document Analyzer",
    page_icon="📚",
    layout="wide"
)

# Configuration
MAX_MESSAGES_PER_CHAT = 60  # Maximum messages (user + assistant) per chat session

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'previous_selected_documents' not in st.session_state:
    st.session_state.previous_selected_documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1

def initialize_rag_system():
    """Initialize the RAG system with dynamic API key."""
    import os
    
    if st.session_state.rag_system is None or st.session_state.get('api_key_changed', False):
        try:
            with st.spinner("Initializing RAG system..."):
                # Update config with current API key if provided
                if st.session_state.gemini_api_key:
                    os.environ['GEMINI_API_KEY'] = st.session_state.gemini_api_key
                    os.environ['USE_FALLBACK_MODEL'] = 'false'
                else:
                    os.environ['USE_FALLBACK_MODEL'] = 'true'
                
                st.session_state.rag_system = RAGSystem()
                st.session_state.api_key_changed = False
            st.success("RAG system initialized successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            st.error("Please check the logs for more details.")
            return False
    return True

def format_chat_history():
    """Format chat history for the LLM context."""
    if "messages" not in st.session_state or len(st.session_state.messages) <= 1:
        return ""
    
    history = []
    for msg in st.session_state.messages[1:]:  # Skip the initial greeting
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Limit each message to avoid context overflow
        if len(content) > 300:
            content = content[:300] + "..."
        history.append(f"{role}: {content}")
    
    return "\n".join(history[-6:])  # Keep last 6 messages (3 Q&A pairs)

def check_document_selection_change():
    """Check if document selection has changed and update chat history accordingly."""
    current_docs = set(st.session_state.selected_documents)
    previous_docs = set(st.session_state.previous_selected_documents)
    
    if current_docs != previous_docs:
        if st.session_state.previous_selected_documents:  # Not the first time
            # Add a system message about document change
            doc_change_msg = f"[Document selection changed from {', '.join(st.session_state.previous_selected_documents)} to {', '.join(st.session_state.selected_documents)}]"
            st.session_state.chat_history += f"\nSystem: {doc_change_msg}\n"
        
        st.session_state.previous_selected_documents = st.session_state.selected_documents.copy()
        return True
    return False

def clear_chat():
    """Clear all chat messages and history to start a new conversation."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I've started a new chat session. What would you like to know about your documents?"}
    ]
    st.session_state.chat_history = ""
    st.success(" New chat session started!")
    st.rerun()



def main():
    # Enhanced header with styling
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0;">📚 RAG Document Analyzer</h1>
        <p style="color: #666; font-size: 1.1rem; margin-top: 0.5rem;">
            Upload your technical documents and ask questions about their content using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the RAG system
    if not initialize_rag_system():
        st.stop()
    
    # Sidebar for document upload and processing
    with st.sidebar:
        # API Configuration Section
        st.markdown("### 🔑 API Configuration")
        
        # Gemini API Key input
        new_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Google Gemini API key for AI responses",
            placeholder="Enter your Gemini API key..."
        )
        
        if new_api_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = new_api_key
            st.session_state.api_key_changed = True
            if new_api_key:
                st.success("✅ API key updated! System will reinitialize.")
            else:
                st.info("ℹ️ Running in demo mode without API key.")
        
        # API Status indicator
        if st.session_state.gemini_api_key:
            st.markdown("🟢 **Status:** API Key Configured")
        else:
            st.markdown("🟡 **Status:** Demo Mode (No API Key)")
        
        st.markdown("---")
        
        # New Chat button (only enabled if documents are uploaded)
        if st.session_state.uploaded_documents:
            if st.button("🆕 New Chat", help="Clear all messages and start a fresh conversation", key="sidebar_new_chat", use_container_width=True):
                clear_chat()
            st.markdown("---")
        
        st.markdown("### 📄 Documents")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD, HTML)",
            type=["pdf", "docx", "txt", "md", "html"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Create documents directory
            docs_dir = "data/documents"
            os.makedirs(docs_dir, exist_ok=True)
            
            # Save uploaded files and track them
            uploaded_file_names = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(docs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_file_names.append(uploaded_file.name)
            
            # Update session state with uploaded documents
            st.session_state.uploaded_documents = list(set(st.session_state.uploaded_documents + uploaded_file_names))
            st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Document selection dropdown (only show if documents are uploaded)
        if st.session_state.uploaded_documents:
            st.markdown("**Select Documents to Search:**")
            selected_docs = st.multiselect(
                "Choose which documents to include in your search",
                options=st.session_state.uploaded_documents,
                default=st.session_state.selected_documents if st.session_state.selected_documents else st.session_state.uploaded_documents,
                help="Select one or more documents to focus your search. Leave empty to search all documents."
            )
            st.session_state.selected_documents = selected_docs if selected_docs else st.session_state.uploaded_documents
            
            # Check for document selection changes
            check_document_selection_change()
            
            # Show current selection status
            # if st.session_state.selected_documents:
            #     st.info(f"🔍 Currently searching in: {', '.join(st.session_state.selected_documents)}")
        
        # Process documents button
        if st.button("🔄 Process Documents", use_container_width=True):
            if not os.path.exists("data/documents") or not os.listdir("data/documents"):
                st.error("Please upload some documents first.")
            else:
                try:
                    with st.spinner("Processing documents... This may take a few minutes."):
                        st.session_state.rag_system.setup_pipeline("data/documents")
                        st.session_state.documents_processed = True
                        st.success("✅ Documents processed successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.error("Please check that your documents are valid and try again.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) system helps you analyze and query 
        your technical documentation using AI.
        
        **How to use:**
        1. Upload your documents using the file uploader above
        2. Click "Process Documents" to index them
        3. Start asking questions about your documents!
        """)
    
    # Main content area
    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process your documents using the sidebar to get started.")
        
        # Enhanced welcome section with cards
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="margin: 0 0 0.5rem 0;">📁 Multi-Format Support</h4>
                <p style="margin: 0; opacity: 0.9;">PDF, DOCX, TXT, Markdown, HTML</p>
            </div>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="margin: 0 0 0.5rem 0;">🤖 AI-Powered Answers</h4>
                <p style="margin: 0; opacity: 0.9;">Natural language questions with citations</p>
            </div>
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white;">
                <h4 style="margin: 0 0 0.5rem 0;">🔍 Smart Search</h4>
                <p style="margin: 0; opacity: 0.9;">Search across all documents simultaneously</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample questions
        st.markdown("### Example questions you can ask:")
        st.code("""
        • "What are the main features of this product?"
        • "How do I configure the authentication system?"
        • "What are the system requirements?"
        • "Summarize the installation process"
        • "What troubleshooting steps are available?"
        """)
        
    else:
        # Chat interface with enhanced styling
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 💬 Ask a question about your documents")
        
        with col2:
            # Collapsible AI Settings
            with st.expander("⚙️ AI Settings", expanded=False):
                st.session_state.temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls randomness: 0.0 = focused, 2.0 = creative"
                )
                
                # Update environment variable for temperature
                os.environ['GEMINI_TEMPERATURE'] = str(st.session_state.temperature)
                
                st.markdown(f"**Current:** {st.session_state.temperature}")
                
                if st.button("🔄 Apply Settings", key="apply_settings", use_container_width=True):
                    st.session_state.api_key_changed = True
                    st.success("Settings applied!")
                    st.rerun()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I've processed your documents and I'm ready to help. What would you like to know?"}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Validate prompt is not empty or just whitespace
            if prompt and prompt.strip():
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Searching through your documents..."):
                            # Format chat history for conversational context
                            chat_history = format_chat_history()
                            
                            # Get response from RAG system with conversational memory and document filter
                            selected_docs = st.session_state.selected_documents if st.session_state.selected_documents else None
                            response = st.session_state.rag_system.ask_question(
                                prompt.strip(), 
                                top_k=30,  # Further increased for maximum coverage
                                selected_documents=selected_docs,
                                chat_history=chat_history
                            )
                            answer = response.get("answer", "I couldn't generate a response.")
                            sources = response.get("sources", [])
                            
                            # Display answer
                            st.markdown(answer)
                            
                            # Display sources if available with enhanced styling
                            if sources:
                                with st.expander("📚 Sources", expanded=False):
                                    for i, source in enumerate(sources, 1):
                                        file_name = source.get('file_name', 'Unknown')
                                        score = source.get('score', 0)
                                        content_preview = source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', '')
                                        
                                        # Color code relevance score
                                        if score >= 0.8:
                                            score_color = "#28a745"  # Green
                                        elif score >= 0.6:
                                            score_color = "#ffc107"  # Yellow
                                        else:
                                            score_color = "#dc3545"  # Red
                                        
                                        st.markdown(f"""
                                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {score_color};">
                                            <h5 style="margin: 0 0 0.5rem 0; color: #333;">📄 {file_name}</h5>
                                            <p style="margin: 0 0 0.5rem 0; color: {score_color}; font-weight: bold;">
                                                Relevance Score: {score:.2f}
                                            </p>
                                            <p style="margin: 0; color: #666; font-style: italic;">
                                                {content_preview}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Add to chat history
                            full_response = answer
                            if sources:
                                full_response += f"\n\n*Found {len(sources)} relevant sources*"
                            
                            st.session_state.messages.append(
                                {"role": "assistant", "content": full_response}
                            )
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

if __name__ == "__main__":
    main()
