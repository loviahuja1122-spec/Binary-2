"""
Embedding Generation Module using Google Gemini API.
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using Google Gemini embedding model.
    Supports batch processing and efficient embedding generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/embedding-001"):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Google API key (defaults to environment variable)
            model_name: Name of the embedding model
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
        self._configure_client()
    
    def _configure_client(self):
        """Configure Google Generative AI client."""
        genai.configure(api_key=self.api_key)
        logger.info(f"Embedding generator initialized with model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="semantic_similarity"
            )
            
            embedding = np.array(result['embedding'], dtype=np.float32)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], 
                                 show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Generating embeddings")
        else:
            iterator = texts
        
        for text in iterator:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed text: {e}")
                # Use zero vector for failed embeddings
                embeddings.append(np.zeros(768, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_chunks(self, chunks: List[Dict], 
                    show_progress: bool = True) -> List[Dict]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            show_progress: Show progress bar
            
        Returns:
            List of chunks with added 'embedding' key
        """
        texts = [chunk['text'] for chunk in chunks]
        
        if show_progress:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        embeddings = self.generate_embeddings_batch(texts, show_progress)
        
        # Attach embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        return chunks
    
    def embed_exam_paper(self, file_path: Path, 
                        exam_metadata: Dict,
                        pdf_processor,
                        show_progress: bool = True) -> List[Dict]:
        """
        Complete pipeline: process PDF and generate embeddings.
        
        Args:
            file_path: Path to exam PDF
            exam_metadata: Metadata about the exam
            pdf_processor: PDFProcessor instance
            show_progress: Show progress bars
            
        Returns:
            List of embedded chunks with metadata
        """
        # Process PDF
        chunks = pdf_processor.process_pdf(file_path, exam_metadata)
        
        # Generate embeddings
        embedded_chunks = self.embed_chunks(chunks, show_progress)
        
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension size
        """
        # Gemini embedding-001 produces 768-dimensional vectors
        return 768
