import os
import json
from typing import List, Dict, Any, Optional
import chromadb

# --- Configuration ---
KB_PATH = os.getenv("KB_PATH", "./chroma_db")

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path=KB_PATH)
    collection = client.get_collection(name="amyloidogenicity")
    KB_READY = True
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    KB_READY = False
    collection = None

# --- KB (Knowledge Base) Tools ---

def kb_status() -> Dict[str, Any]:
    """
    Check knowledge base availability.
    Returns: {ready, path, error?}.
    """
    if KB_READY:
        return {
            "ready": True,
            "path": os.path.abspath(KB_PATH),
            "error": None
        }
    else:
        return {
            "ready": False,
            "path": os.path.abspath(KB_PATH),
            "error": "Vector store not loaded. Ensure KB_PATH is correct and the index exists."
        }

def kb_stats() -> Dict[str, Any]:
    """
    Return basic KB statistics.
    Returns: {docs, chunks, embed_dim, index_type}.
    """
    status = kb_status()
    if not status["ready"]:
        return {"error": status["error"]}

    try:
        count = collection.count()
        return {
            "documents": count,
            "chunks": count,
            "embedding_dimension": 384,
            "index_type": "ChromaDB"
        }
    except Exception as e:
        return {"error": f"Failed to retrieve stats from ChromaDB: {e}"}

# --- RAG Tool ---

def rag_search(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Perform similarity search for relevant chunks in knowledge base about amyloidogenicity.

    :param query: User question.
    :param top_k: Number of chunks to return.
    :param filters: Optional filters (not used with native ChromaDB).
    :return: List of formatted chunks.
    """
    status = kb_status()
    if not status["ready"]:
        print(f"KB Error: {status['error']}")
        return []

    try:
        # Ensure top_k is int (may come as string from LLM)
        top_k = int(top_k) if isinstance(top_k, str) else top_k
        
        # Query using native ChromaDB API
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        formatted_chunks = []
        
        if results['documents'] and len(results['documents']) > 0:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Convert distance to similarity score (lower distance = higher similarity)
                # ChromaDB uses cosine distance, so we convert to similarity
                similarity_score = 1 - distance
                
                formatted_chunks.append({
                    "text": doc_text,
                    "doc_id": metadata.get("doc_id", "N/A"),
                    "page": metadata.get("page", "N/A"),
                    "chunk_id": metadata.get("chunk_id", f"chunk_{i+1}"),
                    "score": similarity_score
                })
        
        return formatted_chunks
        
    except Exception as e:
        print(f"RAG Search Error: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Extra Tool ---

def kb_get_chunk(chunk_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific chunk by ID from the knowledge base.

    :param chunk_id: Chunk identifier.
    :return: Chunk data or error message.
    """
    status = kb_status()
    if not status["ready"]:
        return {"error": status["error"]}
    
    try:
        # Query by chunk_id using metadata filter
        results = collection.get(
            where={"chunk_id": chunk_id}
        )
        
        if results['documents'] and len(results['documents']) > 0:
            doc = results['documents'][0]
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            return {
                "text": doc,
                "doc_id": metadata.get("doc_id", "N/A"),
                "page": metadata.get("page", "N/A"),
                "chunk_id": metadata.get("chunk_id", chunk_id)
            }
        else:
            return {"error": f"Chunk {chunk_id} not found"}
            
    except Exception as e:
        return {"error": f"Failed to retrieve chunk: {e}"}

# --- Export Tools for Qwen-Agent ---

from qwen_agent.tools import BaseTool

class KBStatusTool(BaseTool):
    name = "kb_status"
    description = "Return KB status: {ready, path, error?}."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params: dict, **kwargs):
        return kb_status()

class KBStatsTool(BaseTool):
    name = "kb_stats"
    description = "Return KB stats: {docs, chunks, embed_dim, index_type}."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params: dict, **kwargs):
        return kb_stats()

class RagSearchTool(BaseTool):
    name = "rag_search"
    description = "Retrieve relevant chunks from KB about amyloidogenicity and protein aggregation."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "filters": {"type": "object"}
        },
        "required": ["query"]
    }

    def call(self, params: dict, **kwargs):
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                params = {}
        
        return rag_search(
            query=params.get("query", ""),
            top_k=params.get("top_k", 5),
            filters=params.get("filters")
        )

class KBGetChunkTool(BaseTool):
    name = "kb_get_chunk"
    description = "Retrieve a specific chunk by ID from the knowledge base."
    parameters = {
        "type": "object",
        "properties": {"chunk_id": {"type": "string"}},
        "required": ["chunk_id"]
    }

    def call(self, params: dict, **kwargs):
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                params = {}
        
        return kb_get_chunk(params.get("chunk_id", ""))

TOOLS = [KBStatusTool(), KBStatsTool(), RagSearchTool(), KBGetChunkTool()]
