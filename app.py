"""
Exam Paper Leakage Detection System
Main Streamlit Application

This system detects leaked exam papers by comparing semantic similarity
between uploaded documents and official exam papers using Google Gemini
embeddings and FAISS vector search.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROJECT_ROOT, DATA_DIR, OFFICIAL_PAPERS_DIR, VECTOR_INDEX_DIR,
    DEFAULT_SIMILARITY_THRESHOLD, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD,
    RISK_COLORS, CHUNK_SIZE, CHUNK_OVERLAP
)
from core.embedding import EmbeddingGenerator
from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStore
from core.similarity import SimilarityEngine
from utils.logger import setup_logger
from utils.security import generate_confidential_id, hash_filename

# Setup logging
logger = setup_logger()
logger.info("Application starting...")

# Page configuration
st.set_page_config(
    page_title="Exam Paper Leak Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #4A90D9;
        margin-bottom: 2rem;
    }
    
    .risk-high {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
    }
    
    .risk-medium {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
    }
    
    .risk-low {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
    }
    
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        padding: 1rem;
        background-color: #FFFDE7;
        border-radius: 5px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class ExamLeakDetectionApp:
    """Main application class for Exam Paper Leak Detection System."""
    
    def __init__(self):
        """Initialize application components."""
        self._init_session_state()
        self._init_components()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'embedding_generator' not in st.session_state:
            st.session_state.embedding_generator = None
        if 'pdf_processor' not in st.session_state:
            st.session_state.pdf_processor = PDFProcessor(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
        if 'similarity_engine' not in st.session_state:
            st.session_state.similarity_engine = None
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'threshold' not in st.session_state:
            st.session_state.threshold = DEFAULT_SIMILARITY_THRESHOLD
    
    def _init_components(self):
        """Initialize core components."""
        try:
            # Initialize embedding generator
            if st.session_state.embedding_generator is None:
                st.session_state.embedding_generator = EmbeddingGenerator()
            
            # Initialize vector store
            if st.session_state.vector_store is None:
                st.session_state.vector_store = VectorStore(
                    index_path=VECTOR_INDEX_DIR / "exam_index",
                    embedding_dimension=768
                )
            
            # Initialize similarity engine
            if st.session_state.similarity_engine is None:
                st.session_state.similarity_engine = SimilarityEngine(
                    threshold=st.session_state.threshold
                )
            
            logger.info("All components initialized successfully")
            
        except ValueError as e:
            st.error(f"Configuration Error: {e}")
            st.info("Please set your GOOGLE_API_KEY in the .env file")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Error initializing system: {e}")
    
    def run(self):
        """Run the main application."""
        # Header
        st.markdown('<div class="main-header">🔍 Exam Paper Leak Detection System</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content based on selection
        page = st.session_state.get('current_page', 'Dashboard')
        
        if page == 'Dashboard':
            self._render_dashboard()
        elif page == 'Upload Official':
            self._render_official_upload()
        elif page == 'Upload Suspected':
            self._render_suspected_upload()
        elif page == 'Results':
            self._render_results()
        elif page == 'Flagged Papers':
            self._render_flagged_papers()
        elif page == 'Settings':
            self._render_settings()
        
        # Disclaimer
        self._render_disclaimer()
    
    def _render_sidebar(self):
        """Render sidebar navigation."""
        with st.sidebar:
            st.title("📁 Navigation")
            
            pages = [
                'Dashboard',
                'Upload Official',
                'Upload Suspected',
                'Results',
                'Flagged Papers',
                'Settings'
            ]
            
            current_page = st.radio("Go to", pages, index=pages.index(st.session_state.get('current_page', 'Dashboard')))
            st.session_state.current_page = current_page
            
            st.divider()
            
            # System status
            st.subheader("📊 System Status")
            vector_store = st.session_state.vector_store
            
            if vector_store:
                doc_count = len(vector_store.metadata)
                st.metric("Stored Exams", doc_count)
            else:
                st.metric("Stored Exams", 0)
            
            # Detection history stats
            history = st.session_state.detection_history
            flagged_count = sum(1 for h in history if h.get('is_flagged', False))
            st.metric("Flagged Documents", flagged_count)
            
            st.divider()
            
            # Quick links
            st.subheader("🔗 Quick Links")
            st.markdown("""
            - [GitHub Repository](https://github.com/yourusername/exam-leak-detection-system)
            - [Report Issue](https://github.com/yourusername/exam-leak-detection-system/issues)
            """)
    
    def _render_dashboard(self):
        """Render dashboard page."""
        st.header("📊 Dashboard Overview")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        vector_store = st.session_state.vector_store
        doc_count = len(vector_store.metadata) if vector_store else 0
        
        with col1:
            st.metric("Official Exams", doc_count)
        with col2:
            history = st.session_state.detection_history
            st.metric("Documents Scanned", len(history))
        with col3:
            flagged = sum(1 for h in history if h.get('is_flagged', False))
            st.metric("Flagged Documents", flagged)
        with col4:
            safe = len(history) - flagged
            st.metric("Safe Documents", safe)
        
        st.divider()
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Similarity Distribution")
            if history:
                scores = [h.get('max_similarity', 0) for h in history]
                fig = px.histogram(
                    x=scores, 
                    nbins=20,
                    labels={'x': 'Similarity Score', 'y': 'Count'},
                    color_discrete_sequence=['#4A90D9']
                )
                fig.add_vline(x=st.session_state.threshold, line_dash="dash", 
                             line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No documents scanned yet")
        
        with col_chart2:
            st.subheader("Risk Distribution")
            if history:
                risk_data = []
                for h in history:
                    score = h.get('max_similarity', 0)
                    if score >= HIGH_RISK_THRESHOLD:
                        risk = "High Risk"
                    elif score >= MEDIUM_RISK_THRESHOLD:
                        risk = "Medium Risk"
                    else:
                        risk = "Low Risk"
                    risk_data.append(risk)
                
                risk_counts = pd.Series(risk_data).value_counts()
                fig = px.pie(
                    values=risk_counts.values, 
                    names=risk_counts.index,
                    color_discrete_map={
                        "High Risk": RISK_COLORS['high'],
                        "Medium Risk": RISK_COLORS['medium'],
                        "Low Risk": RISK_COLORS['low']
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No documents scanned yet")
        
        # Recent activity
        st.subheader("📋 Recent Activity")
        if history:
            recent_df = pd.DataFrame(history[-10:])[['timestamp', 'filename', 'max_similarity', 'is_flagged']]
            recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent activity")
    
    def _render_official_upload(self):
        """Render official exam paper upload page."""
        st.header("📤 Upload Official Exam Paper")
        
        with st.form("official_upload_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                exam_name = st.text_input("Exam Name", placeholder="e.g., Final Examination 2024")
                subject = st.text_input("Subject", placeholder="e.g., Mathematics")
            
            with col2:
                date = st.date_input("Exam Date", value=datetime.today())
                grade_level = st.selectbox("Grade Level", 
                                          ["Primary", "Secondary", "High School", "Undergraduate", "Graduate"])
            
            confidential_id = generate_confidential_id()
            st.text_input("Confidential ID", value=confidential_id, disabled=True)
            
            uploaded_file = st.file_uploader("Upload Official PDF", type=['pdf'])
            
            submit_button = st.form_submit_button("Generate Embeddings")
        
        if submit_button:
            if not uploaded_file:
                st.error("Please upload a PDF file")
            elif not exam_name or not subject:
                st.error("Please fill in exam name and subject")
            else:
                self._process_official_upload(
                    uploaded_file, exam_name, subject, 
                    date.strftime("%Y-%m-%d"), confidential_id, grade_level
                )
    
    def _process_official_upload(self, uploaded_file, exam_name, subject, 
                                 date, confidential_id, grade_level):
        """Process official exam paper upload."""
        try:
            with st.spinner("Processing PDF and generating embeddings..."):
                # Save uploaded file
                safe_filename = hash_filename(uploaded_file.name)
                file_path = OFFICIAL_PAPERS_DIR / safe_filename
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Prepare metadata
                exam_metadata = {
                    "exam_name": exam_name,
                    "subject": subject,
                    "date": date,
                    "confidential_id": confidential_id,
                    "grade_level": grade_level
                }
                
                # Generate embeddings
                embedding_gen = st.session_state.embedding_generator
                pdf_processor = st.session_state.pdf_processor
                
                embedded_chunks = embedding_gen.embed_exam_paper(
                    file_path=file_path,
                    exam_metadata=exam_metadata,
                    pdf_processor=pdf_processor
                )
                
                # Store in vector database
                vector_store = st.session_state.vector_store
                vector_store.add_exam_document(
                    exam_name=exam_name,
                    subject=subject,
                    date=date,
                    confidential_id=confidential_id,
                    chunks=embedded_chunks
                )
                
                st.success(f"✅ Successfully stored exam paper: {exam_name}")
                st.balloons()
                
                # Show summary
                st.info(f"""
                **Processing Summary:**
                - File: {uploaded_file.name}
                - Chunks created: {len(embedded_chunks)}
                - Confidential ID: {confidential_id}
                """)
                
        except Exception as e:
            logger.error(f"Error processing official upload: {e}")
            st.error(f"Error processing file: {e}")
    
    def _render_suspected_upload(self):
        """Render suspected document upload page."""
        st.header("📥 Upload Suspected Document")
        
        st.info("Upload a PDF to check for similarity with official exam papers")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Suspected PDF", type=['pdf'], 
                                        key="suspected_upload")
        
        if uploaded_file:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Selected File:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            with col2:
                scan_button = st.button("🔍 Scan Document", type="primary")
            
            if scan_button:
                self._process_suspected_upload(uploaded_file)
    
    def _process_suspected_upload(self, uploaded_file):
        """Process suspected document and check for leaks."""
        try:
            with st.spinner("Analyzing document for potential leaks..."):
                # Save uploaded file temporarily
                safe_filename = hash_filename(uploaded_file.name)
                temp_path = OFFICIAL_PAPERS_DIR / f"suspected_{safe_filename}"
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process PDF
                pdf_processor = st.session_state.pdf_processor
                chunks = pdf_processor.process_pdf(temp_path)
                
                # Generate embeddings
                embedding_gen = st.session_state.embedding_generator
                embedded_chunks = embedding_gen.embed_chunks(chunks, show_progress=False)
                
                # Check similarity
                vector_store = st.session_state.vector_store
                similarity_engine = st.session_state.similarity_engine
                
                results = similarity_engine.check_similarity(
                    embedded_chunks=embedded_chunks,
                    vector_store=vector_store
                )
                
                # Determine if flagged
                is_flagged = results['max_similarity'] >= st.session_state.threshold
                results['is_flagged'] = is_flagged
                results['filename'] = uploaded_file.name
                results['timestamp'] = datetime.now().isoformat()
                
                # Store in history
                st.session_state.detection_history.append(results)
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
                
                # Show results
                st.session_state.current_page = 'Results'
                st.rerun()
                
        except Exception as e:
            logger.error(f"Error processing suspected document: {e}")
            st.error(f"Error analyzing document: {e}")
    
    def _render_results(self):
        """Render similarity results page."""
        st.header("📊 Similarity Results")
        
        history = st.session_state.detection_history
        
        if not history:
            st.info("No documents have been scanned yet. Upload a suspected document to begin.")
            return
        
        # Get latest result
        latest = history[-1]
        
        # Display result prominently
        similarity_score = latest.get('max_similarity', 0)
        threshold = st.session_state.threshold
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Maximum Similarity", f"{similarity_score:.2%}")
        with col2:
            matched_exam = latest.get('matched_exam', 'None')
            st.metric("Matched Exam", matched_exam if matched_exam else "None")
        with col3:
            confidence = latest.get('confidence', 'N/A')
            st.metric("Confidence", confidence)
        
        # Risk indicator
        if similarity_score >= HIGH_RISK_THRESHOLD:
            risk_class = "risk-high"
            risk_label = "🚨 HIGH RISK - LIKELY LEAK"
        elif similarity_score >= MEDIUM_RISK_THRESHOLD:
            risk_class = "risk-medium"
            risk_label = "⚠️ MEDIUM RISK - REVIEW NEEDED"
        else:
            risk_class = "risk-low"
            risk_label = "✅ LOW RISK - APPEARS SAFE"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h3>{risk_label}</h3>
            <p>Similarity Score: {similarity_score:.2%} (Threshold: {threshold:.2%})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed results
        st.subheader("📋 Detailed Analysis")
        
        if latest.get('matches'):
            matches_df = pd.DataFrame(latest['matches'])
            st.dataframe(matches_df, use_container_width=True)
        
        # Similarity by chunk
        if 'chunk_scores' in latest:
            st.subheader("📈 Similarity by Document Section")
            chunk_scores = latest['chunk_scores']
            fig = px.bar(
                x=list(range(len(chunk_scores))),
                y=chunk_scores,
                labels={'x': 'Document Section', 'y': 'Similarity Score'},
                color=[RISK_COLORS['high'] if s >= threshold else 
                       RISK_COLORS['medium'] if s >= MEDIUM_RISK_THRESHOLD else 
                       RISK_COLORS['low'] for s in chunk_scores]
            )
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                         annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        st.subheader("⚡ Actions")
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if latest.get('is_flagged', False):
                if st.button("🚫 Block from Sharing"):
                    st.success("Document has been blocked from sharing")
                if st.button("📢 Report to Authority"):
                    st.success("Report sent to exam authority")
        
        with col_act2:
            if st.button("🔄 Scan Another Document"):
                st.session_state.current_page = 'Upload Suspected'
                st.rerun()
    
    def _render_flagged_papers(self):
        """Render flagged papers list."""
        st.header("⚠️ Flagged Papers")
        
        history = st.session_state.detection_history
        flagged = [h for h in history if h.get('is_flagged', False)]
        
        if not flagged:
            st.success("✅ No flagged papers found. All scanned documents appear safe.")
            return
        
        st.warning(f"Found {len(flagged)} flagged document(s)")
        
        for i, paper in enumerate(flagged):
            with st.expander(f"🚨 {paper.get('filename', 'Unknown')} - {paper.get('max_similarity', 0):.2%}", 
                           expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Similarity Score:** {paper.get('max_similarity', 0):.2%}")
                    st.write(f"**Matched Exam:** {paper.get('matched_exam', 'None')}")
                    st.write(f"**Date Scanned:** {paper.get('timestamp', 'Unknown')}")
                
                with col2:
                    st.write(f"**Confidence:** {paper.get('confidence', 'N/A')}")
                    if paper.get('matches'):
                        st.write(f"**Matched Sections:** {len(paper.get('matches', []))}")
                
                # Action buttons
                col_act1, col_act2, col_act3 = st.columns(3)
                with col_act1:
                    if st.button("Block", key=f"block_{i}"
