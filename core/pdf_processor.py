"""
PDF Processing Module for Exam Paper Leakage Detection System.
Handles text extraction, OCR, and intelligent text chunking.
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

import pdfplumber
from pypdf import PdfReader
import numpy as np

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF text extraction and intelligent chunking.
    Supports both native text PDFs and scanned documents with OCR.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
            
        Raises:
            ValueError: If PDF is empty or unreadable
        """
        logger.info(f"Extracting text from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text_content = ""
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "num_pages": 0,
            "extraction_method": "text"
        }
        
        try:
            # Try pdfplumber first for better text extraction
            with pdfplumber.open(file_path) as pdf:
                metadata["num_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text from page
                        page_text = page.extract_text()
                        
                        if page_text:
                            text_content += page_text + "\n\n"
                        else:
                            # Page might be scanned - need OCR
                            logger.warning(f"Page {page_num + 1} has no extractable text")
                            ocr_text = self._perform_ocr(page)
                            if ocr_text:
                                text_content += ocr_text + "\n\n"
                                metadata["extraction_method"] = "ocr"
                    
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
            
            # Fallback to PyPDF if pdfplumber fails
            if not text_content.strip():
                logger.info("Falling back to PyPDF for text extraction")
                text_content = self._extract_with_pypdf(file_path)
                metadata["extraction_method"] = "pypdf"
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from PDF")
            
            # Generate file hash
            metadata["file_hash"] = self._generate_file_hash(file_path)
            
            logger.info(f"Successfully extracted {len(text_content)} characters from {metadata['num_pages']} pages")
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _extract_with_pypdf(self, file_path: Path) -> str:
        """
        Extract text using PyPDF as fallback.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text string
        """
        text_content = ""
        reader = PdfReader(file_path)
        
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    text_content += text + "\n\n"
            except Exception as e:
                logger.warning(f"PyPDF extraction error: {e}")
                continue
        
        return text_content
    
    def _perform_ocr(self, page) -> str:
        """
        Perform OCR on a scanned PDF page.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            OCR-extracted text
        """
        try:
            import pytesseract
            from PIL import Image
            
            # Convert page to image
            image = page.to_image(resolution=300)
            pil_image = image.original
            
            # Perform OCR
            text = pytesseract.image_to_string(pil_image)
            return text
            
        except ImportError:
            logger.warning("OCR libraries not installed. Install pytesseract for scanned PDFs.")
            return ""
        except Exception as e:
            logger.warning(f"OCR processing error: {e}")
            return ""
    
    def _generate_file_hash(self, file_path: Path) -> str:
        """
        Generate SHA-256 hash of file for security.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks for better embedding coverage.
        
        Args:
            text: Full text content to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Chunking text of {len(text)} characters")
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split into chunks
        chunks = []
        start = 0
        
        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))
            
            # Try to break at sentence boundary
            if end < len(cleaned_text):
                # Find last sentence boundary
                last_period = cleaned_text.rfind('.', start, end)
                last_newline = cleaned_text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk_text = cleaned_text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    "chunk_index": len(chunks),
                    "char_count": len(chunk_text),
                    "start_char": start,
                    "end_char": end
                }
                
                if metadata:
                    chunk_metadata.update({k: v for k, v in metadata.items() 
                                         if k not in ['file_path', 'file_name']})
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start >= end:
                start = end
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text string
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers and headers/footers (simple patterns)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are just page numbers
            if re.match(r'^Page \d+$', line.strip()):
                continue
            # Skip very short lines that might be artifacts
            if len(line.strip()) > 2:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_pdf(self, file_path: Path, exam_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Complete PDF processing pipeline.
        
        Args:
            file_path: Path to PDF file
            exam_metadata: Optional metadata about the exam
            
        Returns:
            List of processed chunks ready for embedding
        """
        # Extract text
        text, pdf_metadata = self.extract_text_from_pdf(file_path)
        
        # Merge metadata
        if exam_metadata:
            pdf_metadata.update(exam_metadata)
        
        # Create chunks
        chunks = self.chunk_text(text, pdf_metadata)
        
        return chunks
