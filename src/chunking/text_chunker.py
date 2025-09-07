"""
Advanced text chunking strategies for optimal retrieval.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)
from loguru import logger

from config import config


@dataclass
class TextChunk:
    """Text chunk with metadata."""
    content: str
    chunk_id: str
    document_id: str
    start_index: int
    end_index: int
    chunk_type: str
    metadata: Dict[str, Any]


class SmartTextChunker:
    """Advanced text chunking with multiple strategies."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunking.max_chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.min_chunk_size = config.chunking.min_chunk_size
        
        # Initialize different splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=config.chunking.separators,
            length_function=len
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size // 4,  # Approximate token count
            chunk_overlap=self.chunk_overlap // 4
        )
        
        # Markdown-specific splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        # HTML-specific splitter
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]
        )
    
    def chunk_document(self, document: Dict[str, Any]) -> List[TextChunk]:
        """
        Chunk a document using the most appropriate strategy.
        
        Args:
            document: Document with content and metadata
            
        Returns:
            List of text chunks
        """
        content = document['content']
        metadata = document['metadata']
        document_id = metadata.hash
        
        # Choose chunking strategy based on document type
        file_type = metadata.file_type.lower()
        
        if file_type in ['.md', '.markdown']:
            chunks = self._chunk_markdown(content, document_id, metadata)
        elif file_type in ['.html', '.htm']:
            chunks = self._chunk_html(content, document_id, metadata)
        elif file_type == '.pdf':
            chunks = self._chunk_pdf(content, document_id, metadata)
        else:
            chunks = self._chunk_generic(content, document_id, metadata)
        
        # Filter out chunks that are too small
        filtered_chunks = [
            chunk for chunk in chunks 
            if len(chunk.content.strip()) >= self.min_chunk_size
        ]
        
        logger.info(f"Created {len(filtered_chunks)} chunks from document {metadata.file_name}")
        return filtered_chunks
    
    def _chunk_markdown(self, content: str, document_id: str, metadata: Any) -> List[TextChunk]:
        """Chunk markdown content preserving structure."""
        chunks = []
        
        try:
            # First split by headers
            header_splits = self.markdown_splitter.split_text(content)
            
            for i, split in enumerate(header_splits):
                # Further split large sections
                if len(split.page_content) > self.chunk_size:
                    sub_chunks = self.recursive_splitter.split_text(split.page_content)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk = TextChunk(
                            content=sub_chunk,
                            chunk_id=f"{document_id}_md_{i}_{j}",
                            document_id=document_id,
                            start_index=0,  # Would need more complex tracking
                            end_index=len(sub_chunk),
                            chunk_type="markdown_section",
                            metadata={
                                **split.metadata,
                                "source_file": metadata.file_name,
                                "file_type": metadata.file_type,
                                "section_index": i,
                                "sub_chunk_index": j
                            }
                        )
                        chunks.append(chunk)
                else:
                    chunk = TextChunk(
                        content=split.page_content,
                        chunk_id=f"{document_id}_md_{i}",
                        document_id=document_id,
                        start_index=0,
                        end_index=len(split.page_content),
                        chunk_type="markdown_section",
                        metadata={
                            **split.metadata,
                            "source_file": metadata.file_name,
                            "file_type": metadata.file_type,
                            "section_index": i
                        }
                    )
                    chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Markdown chunking failed, falling back to generic: {e}")
            chunks = self._chunk_generic(content, document_id, metadata)
        
        return chunks
    
    def _chunk_html(self, content: str, document_id: str, metadata: Any) -> List[TextChunk]:
        """Chunk HTML content preserving structure."""
        chunks = []
        
        try:
            # Split by HTML headers
            header_splits = self.html_splitter.split_text(content)
            
            for i, split in enumerate(header_splits):
                if len(split.page_content) > self.chunk_size:
                    sub_chunks = self.recursive_splitter.split_text(split.page_content)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk = TextChunk(
                            content=sub_chunk,
                            chunk_id=f"{document_id}_html_{i}_{j}",
                            document_id=document_id,
                            start_index=0,
                            end_index=len(sub_chunk),
                            chunk_type="html_section",
                            metadata={
                                **split.metadata,
                                "source_file": metadata.file_name,
                                "file_type": metadata.file_type,
                                "section_index": i,
                                "sub_chunk_index": j
                            }
                        )
                        chunks.append(chunk)
                else:
                    chunk = TextChunk(
                        content=split.page_content,
                        chunk_id=f"{document_id}_html_{i}",
                        document_id=document_id,
                        start_index=0,
                        end_index=len(split.page_content),
                        chunk_type="html_section",
                        metadata={
                            **split.metadata,
                            "source_file": metadata.file_name,
                            "file_type": metadata.file_type,
                            "section_index": i
                        }
                    )
                    chunks.append(chunk)
        except Exception as e:
            logger.warning(f"HTML chunking failed, falling back to generic: {e}")
            chunks = self._chunk_generic(content, document_id, metadata)
        
        return chunks
    
    def _chunk_pdf(self, content: str, document_id: str, metadata: Any) -> List[TextChunk]:
        """Chunk PDF content with special handling for technical documents."""
        chunks = []
        
        # Split by pages first if page markers exist
        page_pattern = r'\n--- Page \d+ ---\n'
        pages = re.split(page_pattern, content)
        
        if len(pages) > 1:
            # Process page by page
            for page_num, page_content in enumerate(pages):
                if not page_content.strip():
                    continue
                    
                page_chunks = self.recursive_splitter.split_text(page_content)
                for chunk_num, chunk_content in enumerate(page_chunks):
                    chunk = TextChunk(
                        content=chunk_content,
                        chunk_id=f"{document_id}_pdf_p{page_num}_c{chunk_num}",
                        document_id=document_id,
                        start_index=0,
                        end_index=len(chunk_content),
                        chunk_type="pdf_page_section",
                        metadata={
                            "source_file": metadata.file_name,
                            "file_type": metadata.file_type,
                            "page_number": page_num,
                            "chunk_index": chunk_num
                        }
                    )
                    chunks.append(chunk)
        else:
            # No page markers, use generic chunking
            chunks = self._chunk_generic(content, document_id, metadata)
        
        return chunks
    
    def _chunk_generic(self, content: str, document_id: str, metadata: Any) -> List[TextChunk]:
        """Generic chunking strategy for any text content."""
        chunks = []
        
        text_chunks = self.recursive_splitter.split_text(content)
        
        for i, chunk_content in enumerate(text_chunks):
            chunk = TextChunk(
                content=chunk_content,
                chunk_id=f"{document_id}_generic_{i}",
                document_id=document_id,
                start_index=0,
                end_index=len(chunk_content),
                chunk_type="generic",
                metadata={
                    "source_file": metadata.file_name,
                    "file_type": metadata.file_type,
                    "chunk_index": i
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_semantic_similarity(self, content: str, document_id: str, metadata: Any) -> List[TextChunk]:
        """
        Advanced chunking based on semantic similarity.
        This would use embeddings to group semantically similar sentences.
        """
        # This is a placeholder for more advanced semantic chunking
        # Would require sentence embeddings and clustering
        return self._chunk_generic(content, document_id, metadata)
    


class ChunkPostProcessor:
    """Post-process chunks for better retrieval."""
    
    def __init__(self):
        pass
    
    def enhance_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Enhance chunks with additional metadata and context.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Enhanced chunks
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Add context from neighboring chunks
            context_before = chunks[i-1].content[-100:] if i > 0 else ""
            context_after = chunks[i+1].content[:100] if i < len(chunks)-1 else ""
            
            # Enhance metadata
            enhanced_metadata = {
                **chunk.metadata,
                "context_before": context_before,
                "context_after": context_after,
                "chunk_position": i,
                "total_chunks": len(chunks),
                "word_count": len(chunk.content.split()),
                "char_count": len(chunk.content)
            }
            
            enhanced_chunk = TextChunk(
                content=chunk.content,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                chunk_type=chunk.chunk_type,
                metadata=enhanced_metadata
            )
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
