import streamlit as st
import os
from pathlib import Path

# MUST be first Streamlit command
st.set_page_config(
    page_title="Amyloidogenicity RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dotenv import load_dotenv
from rag_core import RAGCore
import tempfile

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
print(f"Looking for .env at: {env_path}")
print(f".env exists: {env_path.exists()}")
load_dotenv(dotenv_path=str(env_path), verbose=True)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL")

# Initialize RAGCore with caching
@st.cache_resource
def initialize_rag_core(api_key, model_name):
    """Initialize and cache RAGCore instance."""
    try:
        return RAGCore(api_key=api_key, model_name=model_name)
    except ValueError as e:
        st.error(f"Initialization error: {e}. Check MISTRAL_API_KEY and MISTRAL_MODEL in .env file.")
        return None

rag_system = initialize_rag_core(MISTRAL_API_KEY, MISTRAL_MODEL)

# --- Event Handlers ---

def handle_file_upload(uploaded_file):
    """Save uploaded file to temp directory and return path."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None

def process_documents(file_path):
    """Process documents: load, chunk, embed, and index."""
    if rag_system is None:
        return
        
    try:
        with st.spinner("Processing documents: loading, chunking, and indexing..."):
            chunk_count = rag_system.create_vector_index(file_path)
        
        st.session_state.processed = True
        st.session_state.chunk_count = chunk_count
        st.success(f"Documents processed successfully! Created {chunk_count} chunks.")
        
    except NotImplementedError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error during processing: {e}")

def handle_question_submit(question):
    """Process user question and display answer."""
    if rag_system is None or not st.session_state.get("processed", False):
        st.warning("Please upload and process documents first.")
        return

    if not question:
        st.warning("Please enter a question.")
        return

    with st.spinner("🔍 Searching and generating answer..."):
        try:
            answer, retrieved_chunks = rag_system.answer_question(question)
            
            # Save results to state for display
            st.session_state.last_answer = answer
            st.session_state.last_chunks = retrieved_chunks
            st.session_state.last_question = question
            
        except Exception as e:
            st.error(f"Error generating answer: {e}")

# --- Streamlit Frontend ---

def main():
    """Main Streamlit application."""
    
    st.title("Amyloidogenicity Literature RAG System")
    st.markdown(
        """
        **R**etrieval **A**ugmented **G**eneration system for answering questions about protein aggregation and amyloidogenicity based on your documents.
        
        """
    )

    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.chunk_count = 0
        st.session_state.last_answer = None
        st.session_state.last_chunks = []
        st.session_state.last_question = ""

    # --- Sidebar (Settings and Upload) ---
    with st.sidebar:
        st.header("Settings and Upload")
        
        if rag_system is None:
            st.error("System not initialized. Check MISTRAL_API_KEY and MISTRAL_MODEL in .env file.")
            return

        # 1. Document upload
        uploaded_file = st.file_uploader(
            "Upload PDF document", 
            type=["pdf"], 
            help="Only PDF files are supported."
        )
        
        # 2. Process button
        if uploaded_file is not None:
            temp_file_path = handle_file_upload(uploaded_file)
            
            if st.button("Process and Create Index", key="process_btn"):
                process_documents(temp_file_path)
                
            st.info(f"Uploaded file: **{uploaded_file.name}**")
        else:
            st.session_state.processed = False
            st.session_state.chunk_count = 0
            st.session_state.last_answer = None
            st.session_state.last_chunks = []
            st.session_state.last_question = ""
            st.warning("Waiting for file upload...")

        # 3. System status
        st.markdown("---")
        st.subheader("Status")
        if st.session_state.processed:
            st.success(f"Index ready: {st.session_state.chunk_count} chunks.")
        else:
            st.warning("Index not created.")
            
        st.markdown("---")
        st.caption(f"Using model: **{MISTRAL_MODEL}**")

    # --- Main Area (Question-Answer) ---
    
    st.header("Ask a Question")
    
    # Question form
    with st.form(key="question_form"):
        question = st.text_area(
            "Your question about the document:", 
            value=st.session_state.last_question,
            height=100,
            key="question_input"
        )
        submit_button = st.form_submit_button(
            label="Get Answer", 
            disabled=not st.session_state.processed
        )
        
        if submit_button:
            handle_question_submit(question)

    # --- Results Display ---
    
    if st.session_state.last_answer:
        st.markdown("---")
        st.header("Answer")
        
        # Display answer
        st.info(st.session_state.last_answer)
        
        st.subheader("🔍 Retrieved Context (Chunks)")
        
        # Display retrieved chunks
        with st.expander(f"Show {len(st.session_state.last_chunks)} most relevant chunks"):
            for i, chunk in enumerate(st.session_state.last_chunks):
                st.markdown(f"**Chunk #{i+1}**")
                st.code(chunk, language="text")
                st.markdown("---")

if __name__ == "__main__":
    main()
