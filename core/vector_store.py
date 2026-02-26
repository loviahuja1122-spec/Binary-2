"""
Vector Storage Module using FAISS for efficient similarity search.
"""
import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExamDocument:
    """Represents an embedded exam document."""
    exam_name: str
    subject: str
    date: str
    confidential_id: str
    chunks: List[Dict]
    file_hash: str
    embedding_matrix: np.ndarray


class VectorStore:
    """
    Manages FAISS vector index for exam paper embeddings.
    Supports adding documents, searching, and persistence.
    """
    
    def __init__(self, index_path: Path, embedding_dimension: int = 768):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to store FAISS index
            embedding_dimension: Dimension of embeddings
        """
        self.index_path = index_path
        self.embedding_dimension = embedding_dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.exam_documents: Dict[str, ExamDocument] = {}
        
        # Create index directory
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index
