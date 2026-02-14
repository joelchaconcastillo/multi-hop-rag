"""Hierarchical document chunker for CFR and policy documents."""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a hierarchical document."""

    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None

    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if self.chunk_id is None:
            # Create a simple hash-based ID
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            section = self.metadata.get("section", "unknown")
            self.chunk_id = f"{section}_{content_hash}"


class HierarchicalChunker:
    """Chunks hierarchical documents while preserving structure."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_hierarchy: bool = True,
    ):
        """
        Initialize the hierarchical chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            preserve_hierarchy: Whether to preserve hierarchical markers in metadata
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_hierarchy = preserve_hierarchy

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Chunk text into smaller pieces while preserving hierarchy.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}

        # Extract hierarchical structure
        sections = self._extract_sections(text)
        
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(
                section["content"],
                {**metadata, **section["metadata"]}
            )
            chunks.extend(section_chunks)

        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract hierarchical sections from text.
        
        Handles common hierarchical patterns like:
        - Title 17 → Part 242 → Subpart A → Section 242.100
        - Chapter I → Section 1.1 → Subsection 1.1.1
        """
        sections = []
        
        # Pattern for CFR-style sections (e.g., §242.100, §1.234)
        cfr_pattern = r'§\s*(\d+)\.(\d+)'
        # Pattern for numbered sections (e.g., 1.1, 2.3.4)
        numbered_pattern = r'^(\d+(?:\.\d+)*)\s+'
        # Pattern for titled sections
        title_pattern = r'^(Title|Chapter|Part|Subpart|Section)\s+([^\n]+)'
        
        current_position = 0
        lines = text.split('\n')
        
        current_section = {
            "content": "",
            "metadata": {
                "title": None,
                "chapter": None,
                "part": None,
                "section": None,
                "start_line": 0,
            }
        }
        
        for i, line in enumerate(lines):
            # Check for hierarchical markers
            title_match = re.search(title_pattern, line, re.IGNORECASE)
            cfr_match = re.search(cfr_pattern, line)
            numbered_match = re.search(numbered_pattern, line)
            
            if title_match or cfr_match or numbered_match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                metadata = current_section["metadata"].copy()
                
                if title_match:
                    level, value = title_match.groups()
                    metadata[level.lower()] = value.strip()
                    metadata["section"] = f"{level}_{value.strip()}"
                elif cfr_match:
                    part, section = cfr_match.groups()
                    metadata["part"] = part
                    metadata["section"] = f"{part}.{section}"
                elif numbered_match:
                    section_num = numbered_match.group(1)
                    metadata["section"] = section_num
                
                metadata["start_line"] = i
                
                current_section = {
                    "content": line + "\n",
                    "metadata": metadata
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no sections were found, treat entire document as one section
        if not sections:
            sections = [{
                "content": text,
                "metadata": {
                    "section": "document",
                    "start_line": 0,
                }
            }]
        
        return sections

    def _chunk_section(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk a single section into smaller pieces.

        Args:
            text: The section text to chunk
            metadata: Metadata for this section

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # If section is smaller than chunk_size, return as single chunk
        if len(text) <= self.chunk_size:
            return [DocumentChunk(content=text.strip(), metadata=metadata)]
        
        # Split on paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph would exceed chunk_size
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks)
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata
                ))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last chunk_overlap characters
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata
            ))
        
        return chunks

    def chunk_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Load and chunk a document file.

        Args:
            document_path: Path to the document file
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of DocumentChunk objects
        """
        from pathlib import Path
        
        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Read document based on file type
        if path.suffix.lower() == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif path.suffix.lower() == '.pdf':
            text = self._extract_pdf_text(path)
        elif path.suffix.lower() in ['.doc', '.docx']:
            text = self._extract_docx_text(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        if metadata is None:
            metadata = {}
        
        metadata["source"] = str(path)
        metadata["filename"] = path.name
        
        return self.chunk_text(text, metadata)

    def _extract_pdf_text(self, path) -> str:
        """Extract text from PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install it with: pip install pypdf")

    def _extract_docx_text(self, path) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            doc = Document(path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n\n"
            return text
        except ImportError:
            raise ImportError("python-docx is required for DOCX support. Install it with: pip install python-docx")
