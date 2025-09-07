"""
Document ingestion pipeline supporting multiple formats.
"""
import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import PyPDF2
import pdfplumber
from bs4 import BeautifulSoup
import markdown
from docx import Document
from loguru import logger

from config import config


@dataclass
class DocumentMetadata:
    """Document metadata structure."""
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    created_at: datetime
    modified_at: datetime
    hash: str
    version: str = "1.0"
    source: str = "local"


class DocumentProcessor:
    """Main document processing class."""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.html', '.htm', '.md', '.markdown', '.docx', '.txt'}
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document and extract text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            # Extract text based on file type
            text_content = self._extract_text(file_path)
            
            # Create document structure
            document = {
                'content': text_content,
                'metadata': metadata,
                'processed_at': datetime.now(),
                'chunks': []  # Will be populated by chunking module
            }
            
            logger.info(f"Successfully processed document: {file_path.name}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of processed documents
        """
        documents = []
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    document = self.process_document(str(file_path))
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents
    
    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from file."""
        stat = file_path.stat()
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        return DocumentMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_type=file_path.suffix.lower(),
            file_size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            hash=file_hash
        )
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text content based on file type."""
        file_type = file_path.suffix.lower()
        
        if file_type == '.pdf':
            return self._extract_pdf_text(file_path)
        elif file_type in ['.html', '.htm']:
            return self._extract_html_text(file_path)
        elif file_type in ['.md', '.markdown']:
            return self._extract_markdown_text(file_path)
        elif file_type == '.docx':
            return self._extract_docx_text(file_path)
        elif file_type == '.txt':
            return self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using multiple methods."""
        text_content = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {file_path}: {e2}")
                raise
        
        return text_content.strip()
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            md_content = file.read()
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            
            return soup.get_text()
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        doc = Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        
        return '\n'.join(text_content)
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()


class DocumentVersionManager:
    """Manages document versions and updates."""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or config.processed_dir
        self.version_file = os.path.join(self.storage_path, "document_versions.json")
        
    def check_for_updates(self, document: Dict[str, Any]) -> bool:
        """
        Check if document has been updated since last processing.
        
        Args:
            document: Document with metadata
            
        Returns:
            True if document needs reprocessing
        """
        # Implementation for version checking
        # This would compare file hashes and modification times
        return True  # For now, always reprocess
    
    def update_version_info(self, document: Dict[str, Any]) -> None:
        """Update version information for processed document."""
        # Implementation for updating version tracking
        pass
