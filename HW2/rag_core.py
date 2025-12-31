import os
from typing import List, Tuple
from pypdf import PdfReader
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGCore:
    """RAG system with TF-IDF embeddings and vector store for semantic retrieval."""
    
    def __init__(self, api_key: str, model_name: str):
        """Initialize RAG system with Mistral API key and embedding model."""
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set.")
        
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize TF-IDF vectorizer for embeddings
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Vector store: store chunks and their embeddings
        self.chunks = []
        self.embeddings = None

    def _load_documents(self, file_path: str) -> str:
        """Load PDF and extract text."""
        if not file_path.lower().endswith(".pdf"):
            raise NotImplementedError(f"Loader for {file_path} not implemented.")
        
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return text

    def _split_documents(self, text: str) -> List[str]:
        """Split text into chunks."""
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Combine into chunks of ~1000 chars
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > 1000 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def create_vector_index(self, file_path: str) -> int:
        """
        Process document: load, chunk, generate embeddings, and store in vector index.
        Returns the number of chunks created.
        """
        print(f"Loading document: {file_path}")
        text = self._load_documents(file_path)
        
        print("Splitting into chunks...")
        chunks = self._split_documents(text)
        
        if not chunks:
            return 0

        print(f"Generating embeddings for {len(chunks)} chunks...")
        # Generate TF-IDF embeddings for all chunks
        self.embeddings = self.vectorizer.fit_transform(chunks)
        
        # Store chunks and embeddings in vector store
        self.chunks = chunks
        
        print(f"Vector index created with {len(chunks)} chunks.")
        return len(chunks)

    def _retrieve_chunks(self, question: str, k: int = 4) -> List[str]:
        """
        Retrieve most relevant chunks using semantic similarity (embeddings).
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of most relevant chunks
        """
        if len(self.chunks) == 0 or self.embeddings is None:
            return []
        
        # Step 1: Convert question to embedding using same vectorizer
        question_embedding = self.vectorizer.transform([question])
        
        # Step 2: Compute cosine similarity between question and all chunks
        similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        
        # Step 3: Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top-k chunks
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        return retrieved_chunks

    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """
        Answer question using RAG: retrieval + generation.
        
        RAG Flow:
        1. User asks a question
        2. Question is converted to an embedding (TF-IDF)
        3. Vector store retrieves most relevant chunks using semantic similarity
        4. Chunks are sent to Mistral LLM API along with the question
        5. Mistral generates answer using only the retrieved chunks
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, retrieved_chunks)
        """
        if not self.chunks or self.embeddings is None:
            return "Please upload and process documents first.", []

        print(f"Searching for answer to: {question}")

        # Step 1: Question is already provided by user
        # Step 2: Convert question to embedding and retrieve relevant chunks
        retrieved_chunks = self._retrieve_chunks(question, k=4)
        
        if not retrieved_chunks:
            return "No relevant information found in the documents.", []
        
        # Combine retrieved chunks as context
        context = "\n\n---\n\n".join(retrieved_chunks).strip()

        # Step 3-5: Generate answer with Mistral using retrieved context
        prompt = f"""Use only the provided context to answer the question.
If you cannot find the answer in the context, respond: "The provided documents do not contain information to answer this question."
The answer should be complete but concise.

Context:
{context}

Question: {question}

Answer:"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
            else:
                answer = f"Error: {response.status_code}"
        
        except Exception as e:
            answer = f"Error: {str(e)}"

        return answer, retrieved_chunks
