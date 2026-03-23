# Caching Strategies for RAG Systems

## Overview

Caching is essential for improving RAG system performance, reducing latency, and lowering costs. This document covers various caching strategies for production RAG systems.

## Why Caching Matters

```
Without Caching:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What is RAG?"           Query: "What is RAG?"
        │                                │
        ▼                                ▼
   LLM + Retrieval                  LLM + Retrieval
   (expensive!)                     (expensive again!)

With Caching:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What is RAG?"           Query: "What is RAG?"
        │                                │
        ▼                                ▼
    LLM + Retrieval                    Cache HIT!
    (expensive!)                        (instant)
```

## Cache Layers Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Layer Cache Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────┐                                                         │
│    │ Query Input  │                                                         │
│    └──────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│    ┌──────────────────────────────────────────────┐                         │
│    │         Layer 1: Exact Cache                 │                         │
│    │         (MD5 hash of query)                  │                         │
│    │                                              │                         │
│    │   Query A ──► [HIT] ──► Return               │                         │
│    │   Query B ──► [MISS] ──► Next Layer          │                         │
│    └──────────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│    ┌──────────────────────────────────────────────┐                         │
│    │       Layer 2: Semantic Cache                │                         │
│    │    (Embedding similarity >= 0.95)            │                         │
│    │                                              │                         │
│    │   Similar ──► [HIT] ──► Return               │                         │
│    │   Different ──► [MISS] ──► Next Layer        │                         │
│    └──────────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│    ┌──────────────────────────────────────────────┐                         │
│    │      Layer 3: Retrieval Cache                │                         │
│    │   (Cache documents by query hash)            │                         │
│    │                                              │                         │
│    │   Cached ──► [HIT] ──► Use Docs              │                         │
│    │   Not Cached ──► [MISS] ──► Retrieve         │                         │
│    └──────────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│    ┌──────────────────────────────────────────────┐                         │
│    │      Layer 4: LLM Response Cache             │                         │
│    │    (Cache generated responses)               │                         │
│    │                                              │                         │
│    │   Cached ──► [HIT] ──► Return                │                         │
│    │   Not Cached ──► [MISS] ──► Generate         │                         │
│    └──────────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│    ┌──────────────────────────────────────────────┐                         │
│    │         Full RAG Pipeline                    │                         │
│    │    (Vector Store + LLM Generation)           │                         │
│    │         (Only if all caches miss)            │                         │
│    └──────────────────────┬───────────────────────┘                         │
│                           │                                                 │
│                           ▼                                                 │
│    ┌──────────────────────────────────────────────┐                         │
│    │                Response Output               │                         │
│    └──────────────────────────────────────────────┘                         │
│                                                                             │
│    TTL Settings by Layer:                                                   │
│    • Exact Cache:      1 hour    (fast-changing queries)                    │
│    • Semantic Cache:   1 hour    (similar queries)                          │
│    • Retrieval Cache:  24 hours  (stable knowledge base)                    │
│    • LLM Response:     2 hours   (balance freshness/cost)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Note:** Redis is required for all caching strategies in this document. Install with: `pip install redis`

## Caching Strategies

### 1. Exact Match Caching

Use exact match caching when you have repeated identical queries - it's the simplest and fastest approach. Best for FAQ systems or applications where users ask the same questions repeatedly.

```python
"""
Exact Match Caching
"""

import redis
import json
import hashlib

class ExactMatchCache:
    """Simple exact-match cache for RAG responses."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
    
    def _hash_query(self, query: str) -> str:
        """Create cache key from query."""
        return f"rag:exact:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get(self, query: str) -> dict | None:
        """Get cached response."""
        key = self._hash_query(query)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, query: str, answer: str, sources: list):
        """Cache response."""
        key = self._hash_query(query)
        
        self.redis.setex(
            key,
            self.ttl,
            json.dumps({
                "answer": answer,
                "sources": sources
            })
        )
    
    def invalidate(self, query: str):
        """Invalidate specific cache entry."""
        key = self._hash_query(query)
        self.redis.delete(key)
    
    def clear(self):
        """Clear all cached responses."""
        for key in self.redis.scan_iter("rag:exact:*"):
            self.redis.delete(key)
```

### 2. Semantic Caching

Use semantic caching when users ask the same thing in different ways. It uses embeddings to find cached responses for queries that are semantically similar (e.g., "What is RAG?" and "Explain RAG to me"). Best for chatbots and conversational systems.

```python
"""
Semantic Caching
"""

import numpy as np
from langchain_ollama import OllamaEmbeddings

class SemanticCache:
    """
    Cache responses for semantically similar queries.
    
    Instead of exact matching, we cache based on embedding similarity.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        threshold: float = 0.95,
        ttl: int = 3600
    ):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.threshold = threshold
        self.ttl = ttl
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    def _get_cache_key(self, query_embedding: list) -> str:
        """Find closest cached query using vector search."""
        
        # Scan all cached embeddings
        best_match = None
        best_score = 0
        
        for key in self.redis.scan_iter("rag:semantic:embedding:*"):
            cached_emb_str = self.redis.get(key)
            if not cached_emb_str:
                continue
            
            cached_emb = json.loads(cached_emb_str)
            
            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, cached_emb)
            
            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = key
        
        return best_match.replace("rag:semantic:embedding:", "rag:semantic:response:"), best_score
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str) -> tuple[dict | None, bool, float]:
        """Get cached response and similarity score."""
        
        # Embed query
        query_emb = self.embeddings.embed_query(query)
        
        # Find closest match
        cache_key, score = self._get_cache_key(query_emb)
        
        if cache_key:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached), True, score
        
        return None, False, 0.0
    
    def set(self, query: str, answer: str, sources: list):
        """Cache response with embedding."""
        
        query_emb = self.embeddings.embed_query(query)
        cache_id = hashlib.md5(str(query_emb).encode()).hexdigest()
        
        # Store embedding
        self.redis.setex(
            f"rag:semantic:embedding:{cache_id}",
            self.ttl,
            json.dumps(query_emb)
        )
        
        # Store response
        self.redis.setex(
            f"rag:semantic:response:{cache_id}",
            self.ttl,
            json.dumps({
                "answer": answer,
                "sources": sources,
                "query": query  # Store for debugging
            })
        )
```

### 3. Retrieval Result Caching

Use retrieval result caching when your knowledge base doesn't change often but you process many queries. It caches just the retrieved documents, so you can reuse them with different prompts or LLMs. Best for stable knowledge bases.

```python
"""
Retrieval Result Caching
"""

class RetrievalCache:
    """Cache retrieval results independently."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 86400):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl  # Longer TTL for retrieval results
    
    def _get_key(self, query: str, k: int) -> str:
        """Generate cache key."""
        return f"rag:retrieval:{hashlib.md5(f'{query}:{k}'.encode()).hexdigest()}"
    
    def get(self, query: str, k: int = 4) -> list | None:
        """Get cached documents."""
        key = self._get_key(query, k)
        cached = self.redis.get(key)
        
        if cached:
            docs_data = json.loads(cached)
            # Reconstruct Document objects
            return [Document(**doc) for doc in docs_data]
        return None
    
    def set(self, query: str, documents: list, k: int = 4):
        """Cache retrieved documents."""
        key = self._get_key(query, k)
        
        docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        self.redis.setex(key, self.ttl, json.dumps(docs_data))
```

### 4. LLM Response Caching

Use LLM response caching when you want to save LLM API costs and the same context is used repeatedly. It caches only the generation step, not the retrieval. Best for repeated questions with the same retrieved context.

```python
"""
LLM Response Caching
"""

class LLMResponseCache:
    """Cache LLM responses with context as key."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
    
    def _get_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return f"rag:llm:{hashlib.md5(prompt.encode()).hexdigest()}"
    
    def get(self, prompt: str) -> str | None:
        """Get cached LLM response."""
        key = self._get_key(prompt)
        return self.redis.get(key)
    
    def set(self, prompt: str, response: str):
        """Cache LLM response."""
        key = self._get_key(prompt)
        self.redis.setex(key, self.ttl, response)
```

### 5. Complete Cached RAG Pipeline

```python
"""
Complete Cached RAG Pipeline
"""

class CachedRAGPipeline:
    """Complete RAG pipeline with multiple cache layers."""
    
    def __init__(
        self,
        retriever,
        llm,
        exact_cache: bool = True,
        semantic_cache: bool = True,
        retrieval_cache: bool = True,
        redis_url: str = "redis://localhost:6379"
    ):
        self.retriever = retriever
        self.llm = llm
        
        # Initialize caches
        if exact_cache:
            self.exact_cache = ExactMatchCache(redis_url)
        else:
            self.exact_cache = None
        
        if semantic_cache:
            self.semantic_cache = SemanticCache(redis_url)
        else:
            self.semantic_cache = None
        
        if retrieval_cache:
            self.retrieval_cache = RetrievalCache(redis_url)
        else:
            self.retrieval_cache = None
    
    def query(self, question: str, k: int = 4) -> dict:
        """Query with caching."""
        
        # Step 1: Check exact cache
        if self.exact_cache:
            cached = self.exact_cache.get(question)
            if cached:
                return {
                    "answer": cached["answer"],
                    "sources": cached["sources"],
                    "cached": True,
                    "cache_type": "exact"
                }
        
        # Step 2: Check semantic cache
        if self.semantic_cache:
            cached, is_hit, score = self.semantic_cache.get(question)
            if is_hit:
                return {
                    "answer": cached["answer"],
                    "sources": cached["sources"],
                    "cached": True,
                    "cache_type": "semantic",
                    "similarity": score
                }
        
        # Step 3: Check retrieval cache
        docs = None
        if self.retrieval_cache:
            docs = self.retrieval_cache.get(question, k)
        
        if not docs:
            # Retrieve from vector store
            docs = self.retriever.invoke(question)
            
            # Cache retrieval results
            if self.retrieval_cache:
                self.retrieval_cache.set(question, docs, k)
        
        # Step 4: Generate response
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Check LLM cache
        llm_cache = LLMResponseCache()
        cached_response = llm_cache.get(prompt)
        
        if cached_response:
            answer = cached_response
        else:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            llm_cache.set(prompt, answer)
        
        # Cache full response
        if self.exact_cache:
            self.exact_cache.set(
                question,
                answer,
                [doc.page_content for doc in docs]
            )
        
        return {
            "answer": answer,
            "sources": [doc.page_content for doc in docs],
            "cached": False
        }
```

## Cache Invalidation Strategies

### Time-Based (TTL)

```python
"""
TTL-based Invalidation
"""

# Set appropriate TTLs based on data freshness needs
TTL_SETTINGS = {
    "exact_cache": 3600,        # 1 hour
    "semantic_cache": 3600,     # 1 hour  
    "retrieval_cache": 86400,   # 24 hours
    "llm_cache": 7200,          # 2 hours
}
```

### Event-Based

```python
"""
Event-Based Invalidation
"""

class CacheInvalidator:
    """Invalidate cache when source data changes."""
    
    def __init__(self, cache):
        self.cache = cache
    
    def invalidate_on_document_change(self, document_id: str):
        """Invalidate cache when document is updated/deleted."""
        
        # Option 1: Clear all cache (simple but aggressive)
        self.cache.clear()
        
        # Option 2: Track document IDs in cache (more targeted)
        # Store mapping of document_id -> cache_keys
        # Then invalidate only related cache entries
```

### Hybrid Approach

```python
"""
Hybrid Invalidation
"""

class HybridCacheManager:
    """Combine TTL with manual invalidation."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
    
    def set_with_invalidation(
        self,
        key: str,
        value: dict,
        ttl: int,
        document_ids: list
    ):
        """Set cache with document tracking."""
        
        # Store value
        self.redis.setex(key, ttl, json.dumps(value))
        
        # Track document dependencies
        for doc_id in document_ids:
            # Add key to document's dependency set
            self.redis.sadd(f"doc_deps:{doc_id}", key)
    
    def invalidate_documents(self, document_ids: list):
        """Invalidate all cache entries depending on these documents."""
        
        keys_to_invalidate = set()
        
        for doc_id in document_ids:
            # Get all cache keys depending on this document
            keys = self.redis.smembers(f"doc_deps:{doc_id}")
            keys_to_invalidate.update(keys)
        
        # Invalidate all
        for key in keys_to_invalidate:
            self.redis.delete(key)
            # Also clean up dependency tracking
            for doc_id in document_ids:
                self.redis.srem(f"doc_deps:{doc_id}", key)
```

## Performance Considerations

### Cache Hit Rate Monitoring

```python
"""
Cache Metrics
"""

class CacheMetrics:
    """Track cache performance."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
    
    def get_hit_rate(self, cache_type: str = "exact") -> float:
        """Calculate cache hit rate."""
        
        hits = self.redis.get(f"cache:hits:{cache_type}")
        misses = self.redis.get(f"cache:misses:{cache_type}")
        
        hits = int(hits or 0)
        misses = int(misses or 0)
        
        total = hits + misses
        if total == 0:
            return 0.0
        
        return hits / total
    
    def record_hit(self, cache_type: str = "exact"):
        """Record a cache hit."""
        self.redis.incr(f"cache:hits:{cache_type}")
    
    def record_miss(self, cache_type: str = "exact"):
        """Record a cache miss."""
        self.redis.incr(f"cache:misses:{cache_type}")
```

### When NOT to Cache

| Scenario | Reason |
|----------|--------|
| Personalized results | User-specific, can't share |
| Real-time data | Data changes frequently |
| Complex computations | May not save time |
| First requests | No benefit, just adds latency |

## Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Exact Match** | Fast, simple | Limited hit rate | FAQ, repeated queries |
| **Semantic** | Higher hit rate | Slower lookup | Varied phrasing |
| **Retrieval** | Fast regeneration | Stale context | Stable knowledge bases |
| **LLM** | Save LLM calls | May miss updates | Deterministic responses |
| **Hybrid** | Best of all | Complex | Production systems |

---

## References

### Official Documentation

| Resource | Description |
|----------|-------------|
| [Redis Cache](https://redis.io/docs/) | Redis documentation |
| [LangChain Cache](https://python.langchain.com/docs/modules/cache/) | LangChain caching |
| [LlamaIndex Cache](https://docs.llamaindex.ai/en/stable/module_ides/caching/) | LlamaIndex caching |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [RAG Caching Strategies](https://www.pinecone.io/learn/rag-caching) | Caching patterns |
| [Redis for RAG](https://redis.io/blog/redis-for-rag) | Redis implementation |
| [Semantic Caching](https://blog.langchain.dev/semantic-caching/) | LangChain blog |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [LangChain cache](https://github.com/langchain-ai/langchain/tree/master/libs/langchain-core/langchain_core/caches) | Cache implementations |

---

*Next: [Cost Optimization](cost-optimization.md)*
