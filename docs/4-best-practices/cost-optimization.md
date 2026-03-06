# Cost Optimization

## Overview

RAG systems can become expensive at scale. This document covers strategies to optimize costs while maintaining quality.

## Cost Components

```
RAG Cost Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                         Cost Components                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Retrieval                         Generation                           │
│  ┌─────────────┐                   ┌─────────────┐                      │
│  │ Vector DB   │                   │ LLM API     │                      │
│  │ Hosting     │                   │ Costs       │                      │
│  │ 20-30%      │                   │ 60-70%      │                      │
│  └─────────────┘                   └─────────────┘                      │
│                                                                         │
│  Other                                                                  │
│  ┌─────────────┐                   ┌─────────────┐                      │
│  │ Embedding   │                   │ Compute     │                      │
│  │ Generation  │                   │ (GPU/CPU)   │                      │
│  │ 5-10%       │                   │ 5-10%       │                      │
│  └─────────────┘                   └─────────────┘                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Embedding Cost Optimization

```python
"""
Embedding Cost Optimization
"""

# Strategy 1: Use smaller/faster models
from langchain_community.embeddings import HuggingFaceEmbeddings

# Instead of large model
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Use smaller model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}  # Avoid GPU for small scale
)

# Strategy 2: Batch embedding
def batch_embed(documents: list, batch_size: int = 100):
    """Batch process to reduce API calls."""
    
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Embed batch
        texts = [doc.page_content for doc in batch]
        embeddings = embedding_model.embed_documents(texts)
        
        all_embeddings.extend(embeddings)
        
        print(f"Processed {i + len(batch)}/{len(documents)}")
    
    return all_embeddings

# Strategy 3: Cache embeddings
import hashlib
import pickle
from pathlib import Path

class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_key(self, text: str) -> str:
        """Generate cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str):
        """Get cached embedding."""
        key = self._get_key(text)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None
    
    def set(self, text: str, embedding: list):
        """Cache embedding."""
        key = self._get_key(text)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with open(cache_file, "wb") as f:
            pickle.dump(embedding, f)
```

## 2. Retrieval Cost Optimization

```python
"""
Retrieval Cost Optimization
"""

# Strategy 1: Reduce top_k
# Instead of k=10
results = retriever.invoke(query, k=10)

# Use smaller k
results = retriever.invoke(query, k=3)

# Strategy 2: Use cheaper retrieval first
class TieredRetrieval:
    """Use cheap retrieval first, expensive only if needed."""
    
    def __init__(self):
        self.fast_retriever = BM25Retriever.from_documents(documents)
        self.accurate_retriever = ChromaVectorStore(documents, embeddings)
    
    def retrieve(self, query: str, k: int = 4):
        # Fast retrieval first
        fast_results = self.fast_retriever.invoke(query, k=k*2)
        
        # Check confidence
        if self._high_confidence(fast_results):
            return fast_results[:k]
        
        # Only use expensive if needed
        accurate_results = self.accurate_retriever.invoke(
            query, k=k
        )
        
        return self._merge_results(fast_results, accurate_results, k)
    
    def _high_confidence(self, results) -> bool:
        # Simple threshold
        return len(results) >= 2

# Strategy 3: Limit document size before embedding
MAX_CHUNK_SIZE = 1000

def safe_chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE):
    """Limit chunk size to reduce vector dimensions."""
    return text[:max_size]
```

## 3. Generation Cost Optimization

```python
"""
Generation Cost Optimization
"""

# Strategy 1: Use smaller/faster models

# OPTION A: Use Ollama (local, free) - RECOMMENDED
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")  # Free, runs locally

# OPTION B: Use OpenAI (cloud API)
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini")  # Cheaper than gpt-4o

# Strategy 2: Reduce context
def truncate_context(context: str, max_tokens: int = 4000):
    """Truncate context to fit in window."""
    
    # Rough token estimation
    tokens = context.split()
    
    if len(tokens) > max_tokens * 0.75:  # ~4 chars per token
        return " ".join(tokens[:int(max_tokens * 0.75)])
    
    return context

# Strategy 3: Selective context
def select_relevant_context(results: list, query: str) -> str:
    """Select only relevant parts of context."""
    
    # Simple: take first few results
    return "\n\n".join([
        doc.page_content for doc in results[:2]
    ])

# Strategy 4: Cache LLM responses
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Cache LLM responses."""
    
    llm = ChatOpenAI(model=model)
    return llm.invoke(prompt)
```

## 4. Vector Database Cost

```python
"""
Vector Database Cost Optimization
"""

# Strategy 1: Use open-source databases
# Instead of Pinecone (paid)

# Use Chroma (free, local)
vectorstore = Chroma.from_documents(documents, embeddings)

# Or use pgvector (if you have Postgres)
vectorstore = PGVector.from_documents(
    documents, 
    embedding=embeddings,
    connection_string="postgresql://..."
)

# Strategy 2: Optimize index
def optimize_index(vectorstore):
    """Reduce index size and improve speed."""
    
    # Reduce dimensions if using PCA
    # Or use quantization
    
    # For Pinecone
    vectorstore.describe_index_stats()
    
    # Check and optimize

# Strategy 3: Delete unused vectors
def cleanup_unused_vectors(vectorstore, threshold_days: int = 30):
    """Remove old/unused vectors."""
    
    # Identify old vectors
    old_vectors = []
    
    # Delete in batches
    for batch in batches(old_vectors, 1000):
        vectorstore.delete(ids=batch)
```

## 5. Complete Cost-Optimized RAG

```python
"""
Complete Cost-Optimized RAG Pipeline

This is a COST-OPTIMIZED RAG, so we use FREE local models by default (Ollama).
For paid alternatives, simply switch to OpenAI.
"""

class CostOptimizedRAG:
    """RAG with maximum cost optimization - uses FREE local models by default."""
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        provider: str = "ollama",  # "ollama" (free) or "openai" (paid)
        use_cache: bool = True,
        use_tiered_retrieval: bool = True
    ):
        self.provider = provider
        
        # Initialize embeddings based on provider
        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(model=embedding_model)
        else:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize LLM based on provider
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model=llm_model)
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=llm_model)
        
        # Setup caching
        self.use_cache = use_cache
        if use_cache:
            self.embedding_cache = EmbeddingCache()
            self.query_cache = {}
        
        # Tiered retrieval
        self.vectorstore = None  # Will be set when documents are added
        if use_tiered_retrieval:
            self.retriever = TieredRetrieval()
    
    def add_documents(self, documents: list):
        """Add documents to the vector store."""
        from langchain_community.vectorstores import Chroma
        
        self.vectorstore = Chroma.from_documents(
            documents, 
            self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(k=3)
    
    def query(self, question: str) -> dict:
        """Query with cost optimization."""
        
        # Check cache
        if self.use_cache and question in self.query_cache:
            return self.query_cache[question]
        
        # Retrieve (cheap)
        docs = self.retriever.invoke(question)
        
        # Truncate context
        context = "\n\n".join([d.page_content for d in docs])[:4000]
        
        # Generate
        if self.provider == "ollama":
            # Ollama uses .invoke()
            response = self.llm.invoke(context + "\n\nQuestion: " + question)
            answer = response.content if hasattr(response, 'content') else str(response)
        else:
            # OpenAI uses .invoke() with messages format
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([
                HumanMessage(content=context + "\n\nQuestion: " + question)
            ])
            answer = response.content if hasattr(response, 'content') else str(response)
        
        # Cache result
        if self.use_cache:
            self.query_cache[question] = {"answer": answer, "sources": docs}
        
        return {"answer": answer, "sources": docs}
    
    def get_cost_estimate(self, question: str) -> dict:
        """Estimate cost for a query."""
        
        if self.provider == "ollama":
            # Ollama is FREE
            return {
                "embedding": 0.0,
                "retrieval": 0.0,
                "generation": 0.0,
                "total": 0.0,
                "note": "Using Ollama - completely FREE!"
            }
        
        # OpenAI costs (approximate)
        embedding_cost = len(question) / 1000 * 0.00002  # $0.02/1M tokens
        retrieval_cost = 0.0001  # Fixed
        generation_cost = len(question) / 1000 * 0.00015  # gpt-4o-mini
        
        return {
            "embedding": embedding_cost,
            "retrieval": retrieval_cost,
            "generation": generation_cost,
            "total": sum([embedding_cost, retrieval_cost, generation_cost])
        }


# Usage with FREE Ollama (RECOMMENDED)
rag = CostOptimizedRAG(provider="ollama")  # FREE!
rag.add_documents(documents)
result = rag.query("What is RAG?")

# Usage with paid OpenAI (alternative)
# rag = CostOptimizedRAG(provider="openai", llm_model="gpt-4o-mini")
```

## Cost Comparison

| Optimization | Quality Impact | Cost Savings |
|--------------|----------------|---------------|
| **Use Ollama** | Minimal (-10-20%) | 95-100% |
| Smaller embedding model | Minimal (-5%) | 50-70% |
| Smaller LLM model | Moderate (-10%) | 80-90% |
| Smaller k | Minimal | 30-50% |
| Caching | None | 40-60% |
| Tiered retrieval | Minimal | 30-40% |
| Truncate context | Minimal | 20-30% |

## Free Option: Ollama

The biggest cost savings is using **Ollama** (local models) instead of OpenAI:

```python
# OpenAI (paid)
llm = ChatOpenAI(model="gpt-4")  # ~$0.01-0.03/query

# Use Ollama (free, local)
llm = ChatOllama(model="llama3.2")  # Free!

# Same for embeddings
# OpenAI: ~$0.00002/1K tokens
# Ollama: Free
```

See [Providers](../3-technical/providers.md) for setup instructions.

---

*Next: [Scaling Patterns](scaling-patterns.md)*
