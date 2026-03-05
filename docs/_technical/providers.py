"""
RAG Provider Module

A simple wrapper for RAG implementations supporting multiple providers.
"""

from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGProvider:
    """Simple RAG provider supporting Ollama and OpenAI."""
    
    def __init__(
        self,
        provider: str = "ollama",
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG provider.
        
        Args:
            provider: "ollama" or "openai"
            llm_model: Model name for the LLM
            embedding_model: Model name for embeddings
            chunk_size: Text chunk size
            chunk_overlap: Text chunk overlap
        """
        self.provider = provider
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._llm = None
        self._embeddings = None
        self._vectorstore = None
        self._retriever = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        
    def _init_ollama(self):
        """Initialize Ollama models."""
        self._llm = ChatOllama(model=self.llm_model)
        self._embeddings = OllamaEmbeddings(model=self.embedding_model)
        
    def _init_openai(self):
        """Initialize OpenAI models."""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            self._llm = ChatOpenAI(model=self.llm_model)
            self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
    
    def _ensure_initialized(self):
        """Ensure models are initialized."""
        if self._llm is None:
            if self.provider == "ollama":
                self._init_ollama()
            elif self.provider == "openai":
                self._init_openai()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document strings
        """
        self._ensure_initialized()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        docs = [Document(page_content=doc) for doc in documents]
        split_docs = text_splitter.split_documents(docs)
        
        self._vectorstore = InMemoryVectorStore.from_documents(
            split_docs,
            embedding=self._embeddings
        )
        self._retriever = self._vectorstore.as_retriever()
        
    def query(
        self,
        question: str,
        use_cache: bool = True,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            use_cache: Whether to use caching
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer, sources, and cached flag
        """
        if not question:
            raise ValueError("Question cannot be empty")
        
        # Check cache
        cache_key = question.lower().strip()
        if use_cache and cache_key in self._cache:
            result = self._cache[cache_key].copy()
            result["cached"] = True
            return result
        
        self._ensure_initialized()
        
        # Retrieve documents
        if self._retriever is None:
            raise ValueError("No documents added. Call add_documents() first.")
        
        docs = self._retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        # Generate response
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
        
        answer = self._llm.invoke(prompt).content
        
        # Prepare result
        result = {
            "answer": answer,
            "sources": [{"content": d.page_content} for d in docs],
            "cached": False
        }
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result.copy()
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a generative AI model.",
        "RAG helps reduce hallucinations by grounding responses in retrieved context.",
        "Benefits of RAG include: fresh knowledge, reduced hallucinations, traceability, and cost-effective updates.",
        "Vector search works by converting text into embeddings and finding similar vectors.",
        "Embeddings are numerical representations of text that capture semantic meaning."
    ]
    
    # Using Ollama (default)
    print("Using Ollama provider:")
    rag = RAGProvider(provider="ollama")
    rag.add_documents(documents)
    result = rag.query("What is RAG?")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Cached: {result['cached']}")
