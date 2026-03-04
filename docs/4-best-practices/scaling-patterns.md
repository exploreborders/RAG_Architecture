# Scaling Patterns

## Overview

Scaling RAG systems for production requires addressing performance, reliability, and cost. This document covers patterns for scaling to millions of documents and thousands of users.

## Scaling Challenges

```
RAG Scaling Challenges:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                     Scaling Dimensions                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   Data Scale    │  │   Query Load    │  │  Latency        │          │
│  │                 │  │                 │  │  Requirements   │          │
│  │ • Millions of   │  │ • Thousands of  │  │                 │          │
│  │   documents     │  │   concurrent    │  │ • <100ms for    │          │
│  │ • TB of data    │  │   users         │  │   real-time     │          │
│  │ • Frequent      │  │ • Burst         │  │ • <1s for       │          │
│  │   updates       │  │   traffic       │  │   batch         │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Horizontal Scaling

```python
"""
Horizontal Scaling: Multiple RAG Instances
"""

import threading
from queue import Queue

class LoadBalancedRAG:
    """Distribute queries across multiple RAG instances."""
    
    def __init__(self, instances: list):
        self.instances = instances
        self.current = 0
        self.lock = threading.Lock()
    
    def query(self, question: str) -> dict:
        """Round-robin across instances."""
        
        with self.lock:
            instance = self.instances[self.current]
            self.current = (self.current + 1) % len(self.instances)
        
        return instance.query(question)
    
    def query_with_fallback(self, question: str) -> dict:
        """Try instances with fallback."""
        
        errors = []
        
        for instance in self.instances:
            try:
                return instance.query(question)
            except Exception as e:
                errors.append(str(e))
                continue
        
        raise Exception(f"All instances failed: {errors}")
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    spec:
      containers:
      - name: rag-api
        image: rag-api:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 2. Caching Strategies

```python
"""
Multi-Level Caching
"""

import hashlib
import json
from functools import lru_cache
import redis

class CacheManager:
    """Multi-level cache for RAG."""
    
    def __init__(self):
        # L1: In-memory cache
        self.l1_cache = {}
        
        # L2: Redis cache
        self.redis = redis.Redis(host='localhost', port=6379)
    
    def _make_key(self, query: str, context: dict = None) -> str:
        """Create cache key."""
        
        key_data = {"query": query, "context": context or {}}
        return hashlib.sha256(json.dumps(key_data).encode()).hexdigest()
    
    def get(self, query: str, context: dict = None) -> str:
        """Get cached result."""
        
        key = self._make_key(query, context)
        
        # Check L1
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2
        cached = self.redis.get(key)
        if cached:
            self.l1_cache[key] = cached  # Promote to L1
            return cached
        
        return None
    
    def set(self, query: str, result: str, context: dict = None, ttl: int = 3600):
        """Cache result."""
        
        key = self._make_key(query, context)
        
        # Store in both levels
        self.l1_cache[key] = result
        self.redis.setex(key, ttl, result)
    
    def invalidate(self, query: str):
        """Invalidate cache entry."""
        
        key = self._make_key(query)
        self.l1_cache.pop(key, None)
        self.redis.delete(key)


# Usage in RAG
class CachedRAG:
    """RAG with caching."""
    
    def __init__(self, rag, cache: CacheManager):
        self.rag = rag
        self.cache = cache
    
    def query(self, question: str, use_cache: bool = True) -> dict:
        """Query with caching."""
        
        if use_cache:
            cached = self.cache.get(question)
            if cached:
                return {"result": cached, "cached": True}
        
        result = self.rag.query(question)
        
        self.cache.set(question, result["result"])
        
        return {**result, "cached": False}
```

## 3. Index Partitioning

```python
"""
Partition Large Indexes
"""

class PartitionedRAG:
    """Divide data across multiple indexes."""
    
    def __init__(self, partitions: dict):
        """
        partitions: {
            "2024": vectorstore_2024,
            "2025": vectorstore_2025,
            "general": vectorstore_general
        }
        """
        self.partitions = partitions
    
    def query(self, question: str) -> dict:
        """Query relevant partitions."""
        
        # Determine relevant partitions (could use LLM or rules)
        partitions_to_query = self._select_partitions(question)
        
        # Query each partition
        all_results = []
        for partition_name in partitions_to_query:
            results = self.partitions[partition_name].similarity_search(
                question, k=4
            )
            all_results.extend(results)
        
        # Re-rank combined results
        return self._rerank(question, all_results)
    
    def _select_partitions(self, question: str) -> list:
        """Select relevant partitions."""
        
        # Simple rule-based
        if "2024" in question:
            return ["2024", "general"]
        elif "2025" in question:
            return ["2025", "general"]
        
        return list(self.partitions.keys())
    
    def _rerank(self, question: str, results: list) -> dict:
        """Re-rank across partitions."""
        
        # Simple re-ranking logic
        return {"results": results[:4], "total": len(results)}
```

## 4. Async Processing

```python
"""
Async RAG for High Throughput
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAG:
    """Process queries asynchronously."""
    
    def __init__(self, rag, max_workers: int = 10):
        self.rag = rag
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def query(self, question: str) -> dict:
        """Async query."""
        
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self.executor,
            self.rag.query,
            question
        )
        
        return result
    
    async def batch_query(self, questions: list) -> list:
        """Process multiple queries concurrently."""
        
        tasks = [self.query(q) for q in questions]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def query_sync(self, question: str) -> dict:
        """Synchronous interface."""
        
        return asyncio.run(self.query(question))
```

## 5. Data Update Strategies

```python
"""
Incremental Index Updates
"""

class IncrementalIndexer:
    """Update index incrementally."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def add_documents(self, documents: list, batch_size: int = 100):
        """Add documents in batches."""
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Add to vector store
            self.vectorstore.add_documents(batch)
            
            print(f"Indexed batch {i//batch_size + 1}")
    
    def update_documents(self, updates: dict):
        """
        Update existing documents.
        
        updates: {doc_id: new_content}
        """
        
        for doc_id, new_content in updates.items():
            # Delete old
            self.vectorstore.delete([doc_id])
            
            # Add new
            self.vectorstore.add_documents([new_content])
    
    def incremental_update(self, new_documents: list, change_detection: callable):
        """Detect changes and update incrementally."""
        
        # Find new documents
        existing_ids = set(self.vectorstore.get()["ids"])
        new_ids = set()
        
        for doc in new_documents:
            doc_id = doc.metadata.get("id")
            new_ids.add(doc_id)
            
            if doc_id not in existing_ids:
                # New document
                self.vectorstore.add_documents([doc])
            
            elif change_detection(doc):
                # Changed document
                self.update_documents({doc_id: doc})
        
        # Handle deletions
        deleted = existing_ids - new_ids
        if deleted:
            self.vectorstore.delete(list(deleted))
```

## 6. Monitoring & Observability

```python
"""
RAG Monitoring
"""

import time
from dataclasses import dataclass
from typing import List

@dataclass
class RAGMetrics:
    """Track RAG performance metrics."""
    
    total_queries: int = 0
    cache_hits: int = 0
    avg_latency_ms: float = 0
    error_count: int = 0
    retrieval_latency_ms: float = 0
    generation_latency_ms: float = 0
    
    def record_query(self, latency_ms: float, cached: bool, 
                    retrieval_ms: float, generation_ms: float):
        """Record a query."""
        
        self.total_queries += 1
        
        if cached:
            self.cache_hits += 1
        
        # Running average
        n = self.total_queries
        self.avg_latency_ms = (
            (self.avg_latency_ms * (n-1) + latency_ms) / n
        )
        self.retrieval_latency_ms = (
            (self.retrieval_latency_ms * (n-1) + retrieval_ms) / n
        )
        self.generation_latency_ms = (
            (self.generation_latency_ms * (n-1) + generation_ms) / n
        )
    
    def get_stats(self) -> dict:
        """Get current stats."""
        
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hits / max(1, self.total_queries),
            "avg_latency_ms": self.avg_latency_ms,
            "avg_retrieval_ms": self.retrieval_latency_ms,
            "avg_generation_ms": self.generation_latency_ms,
            "errors": self.error_count
        }


# Usage
metrics = RAGMetrics()

class MonitoredRAG:
    """RAG with monitoring."""
    
    def __init__(self, rag):
        self.rag = rag
        self.metrics = metrics
    
    def query(self, question: str) -> dict:
        """Query with timing."""
        
        start = time.time()
        
        # Retrieval timing
        t0 = time.time()
        retrieval_results = self.rag.retrieve(question)
        t_retrieval = (time.time() - t0) * 1000
        
        # Generation timing
        t1 = time.time()
        result = self.rag.generate(question, retrieval_results)
        t_generation = (time.time() - t1) * 1000
        
        total_ms = (time.time() - start) * 1000
        
        self.metrics.record_query(
            total_ms,
            result.get("cached", False),
            t_retrieval,
            t_generation
        )
        
        return result
```

---

*Next: [Cost Optimization](cost-optimization.md)*
