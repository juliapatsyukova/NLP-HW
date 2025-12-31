#!/usr/bin/env python3
"""
Build ChromaDB index from PDF files in sample_docs/
Uses native ChromaDB API (not LangChain) for better control.
Run this ONCE before using the agent.
"""

import os
import glob
import shutil
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# Configuration
SAMPLE_DOCS_PATH = "./sample_docs"
KB_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def build_kb():
    """Build ChromaDB index from PDFs using native ChromaDB API."""
    
    print("=" * 60)
    print("BUILDING CHROMADB INDEX FROM PDFs")
    print("=" * 60)
    
    # Find PDF files
    pdf_files = glob.glob(os.path.join(SAMPLE_DOCS_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"ERROR: No PDF files found in {SAMPLE_DOCS_PATH}")
        return False
    
    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {os.path.basename(pdf)}")
    
    # Extract and chunk documents
    documents = []
    metadatas = []
    ids = []
    
    for pdf_path in pdf_files:
        pdf_name = Path(pdf_path).stem
        print(f"\nProcessing: {pdf_name}...")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"  WARNING: No text extracted from {pdf_name}")
            continue
        
        # Split into chunks
        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")
        
        # Add to documents
        for i, chunk in enumerate(chunks):
            # Extract page number from chunk if available
            page_num = 1
            if "--- Page" in chunk:
                try:
                    page_str = chunk.split("--- Page ")[1].split(" ---")[0]
                    page_num = int(page_str)
                except:
                    pass
            
            doc_id = f"{pdf_name}_chunk_{i:04d}"
            documents.append(chunk)
            metadatas.append({
                "doc_id": pdf_name,
                "page": page_num,
                "chunk_id": f"chunk_{i:04d}"
            })
            ids.append(doc_id)
    
    if not documents:
        print("\nERROR: No documents to index")
        return False
    
    print(f"\nTotal documents to index: {len(documents)}")
    
    # Remove old KB if exists
    if os.path.exists(KB_PATH):
        print(f"\nRemoving old KB at {KB_PATH}...")
        shutil.rmtree(KB_PATH)
    
    # Create new ChromaDB with persistent storage
    print(f"\nCreating ChromaDB at {KB_PATH}...")
    try:
        # Initialize ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path=KB_PATH)
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="amyloidogenicity",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"  Collection created: {collection.name}")
        
        # Add documents to collection
        print(f"  Adding {len(documents)} documents...")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        # Verify documents were added
        count = collection.count()
        print(f"  ✓ Documents added: {count}")
        
        if count == 0:
            print("  ERROR: No documents were added to collection!")
            return False
        
        print(f"\n✓ ChromaDB created successfully")
        print(f"✓ Indexed {count} chunks")
        print(f"✓ KB saved to {KB_PATH}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = build_kb()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS! KB is ready.")
        print("Now run: python agent_app.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED! Check errors above.")
        print("=" * 60)
