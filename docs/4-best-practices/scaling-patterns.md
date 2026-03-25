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

**Concept**: When a single RAG instance can't handle your query load, you can run multiple instances behind a load balancer. Each instance maintains its own vector database connection and LLM client, distributing the load across machines.

**Why it helps**:
- Handles more concurrent users without slowdown
- Provides redundancy (if one instance fails, others continue)
- Allows gradual scaling (add instances as needed)

**When to use**:
- More than 50-100 concurrent users
- Need high availability (no single point of failure)
- CPU/memory on single instance exceeds 80% under load
- Response times degrading under load

**Example**:
- Single instance: handles 100 users at 500ms avg latency
- 3 instances behind load balancer: handles 300 users at 500ms avg latency
- If one instance fails, requests route to remaining 2 instances

```python
"""
Horizontal Scaling: Multiple RAG Instances
"""

class LoadBalancedRAG:
    """Distribute queries across multiple RAG instances with health awareness."""
    
    def __init__(self, instances: list):
        self.instances = instances
        self.current = 0
        # Track instance health
        self.health_status = {id(inst): True for inst in instances}
    
    def query(self, question: str) -> dict:
        """Round-robin across healthy instances."""
        
        # Find next healthy instance
        attempts = 0
        while attempts < len(self.instances):
            instance = self.instances[self.current]
            self.current = (self.current + 1) % len(self.instances)
            
            if self.health_status.get(id(instance), False):
                return instance.query(question)
            attempts += 1
        
        # All unhealthy, try any
        return self.instances[0].query(question)
    
    def mark_unhealthy(self, instance):
        """Mark an instance as unhealthy."""
        self.health_status[id(instance)] = False
    
    def mark_healthy(self, instance):
        """Mark an instance as healthy."""
        self.health_status[id(instance)] = True
    
    def query_with_fallback(self, question: str) -> dict:
        """Try instances with fallback."""
        
        errors = []
        
        for instance in self.instances:
            if not self.health_status.get(id(instance), True):
                continue
            try:
                return instance.query(question)
            except Exception as e:
                errors.append(str(e))
                self.mark_unhealthy(instance)
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

**Concept**: Caching stores previously computed results so identical or similar queries can be answered without re-running the expensive retrieval and generation steps. Multi-level caching puts fast, small caches close to the application and slower, larger caches further away.

**Why it helps**:
- Identical queries served instantly from cache (< 1ms vs 500ms+)
- Reduces LLM API costs (fewer generation calls)
- Reduces vector DB load
- Improves consistency of response times

**Multi-Level Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                      Request Flow                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Query ──► L1 Cache (Memory) ──► L2 Cache (Redis) ──► RAG  │
│              ~1ms hit                ~10ms hit    ~500ms    │
│                                                             │
│   Cache Hit: Return immediately                             │
│   Cache Miss: Process full RAG, store in both levels        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**When to use**:
- High volume of repeated queries
- Questions that users ask frequently (FAQ-style)
- Want to reduce LLM API costs
- Need consistent, fast response times

**Example**:
- Without cache: 1000 queries/min → 1000 LLM calls, 500ms avg latency
- With 80% cache hit rate: 200 LLM calls, ~100ms avg latency
- Cost savings: 80% reduction in LLM API costs

```python
"""
Multi-Level Caching
"""

import hashlib
import json
import redis
from collections import OrderedDict

class CacheManager:
    """Multi-level cache for RAG with LRU in-memory cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", l1_maxsize: int = 1000):
        # L1: In-memory LRU cache (faster, limited size)
        self.l1_cache = OrderedDict()
        self.l1_maxsize = l1_maxsize
        
        # L2: Redis cache (slower, persistent, larger)
        self.redis = redis.from_url(redis_url, decode_responses=True)
    
    def _make_key(self, query: str, context: dict = None) -> str:
        """Create cache key from query and optional context."""
        
        key_data = {"query": query.lower().strip(), "context": context or {}}
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, context: dict = None) -> str | None:
        """Get cached result from L1 or L2."""
        
        key = self._make_key(query, context)
        
        # Check L1 (fastest)
        if key in self.l1_cache:
            # Move to end (most recently used)
            self.l1_cache.move_to_end(key)
            return self.l1_cache[key]
        
        # Check L2
        cached = self.redis.get(key)
        if cached:
            # Promote to L1
            self._l1_set(key, cached)
            return cached
        
        return None
    
    def _l1_set(self, key: str, value: str):
        """Set value in L1 cache with LRU eviction."""
        self.l1_cache[key] = value
        self.l1_cache.move_to_end(key)
        
        # Evict oldest if over size
        if len(self.l1_cache) > self.l1_maxsize:
            self.l1_cache.popitem(last=False)
    
    def set(self, query: str, result: str, context: dict = None, ttl: int = 3600):
        """Cache result in both levels."""
        
        key = self._make_key(query, context)
        
        # Store in L1
        self._l1_set(key, result)
        
        # Store in L2 with TTL
        self.redis.setex(key, ttl, result)
    
    def invalidate(self, query: str, context: dict = None):
        """Invalidate cache entry from both levels."""
        
        key = self._make_key(query, context)
        self.l1_cache.pop(key, None)
        self.redis.delete(key)


# Usage in RAG
class CachedRAG:
    """RAG with multi-level caching."""
    
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
        result_text = result.get("result", result) if isinstance(result, dict) else result
        
        self.cache.set(question, result_text)
        
        return {"result": result_text, "cached": False}
```

## 3. Index Partitioning

**Concept**: Instead of storing all documents in one large vector index, partition your data into smaller, focused indexes. Each partition contains a subset of documents (e.g., by topic, date, or category), allowing queries to search only relevant partitions.

**Why it helps**:
- Reduces search space per query (faster retrieval)
- Enables domain-specific indexing (better embedding match)
- Allows independent scaling of partitions
- Simplifies data management (easier updates, backups)

**Partitioning Strategies**:

| Strategy | Best For | Example |
|----------|----------|---------|
| **By topic** | Multi-domain knowledge bases | docs, blog, support |
| **By date** | Time-sensitive data | 2024, 2025, archives |
| **By access** | Permission-based content | public, internal, admin |
| **By size** | Large datasets | chunks of 1M vectors |
| **By geography** | Regional data | US, EU, APAC |

**When to use**:
- More than 10M vectors in a single index
- Documents span distinct domains or topics
- Different teams manage different document collections
- Need to scale specific partitions independently

**Example**:
```
Unpartitioned: 10M vectors → search all 10M
Partitioned by topic:
  - docs: 3M vectors → search 3M
  - blog: 2M vectors → search 2M  
  - support: 1M vectors → search 1M
  
Result: ~3x faster retrieval for targeted queries
```

```python
"""
Partition Large Indexes
"""

class PartitionedRAG:
    """Divide data across multiple indexes."""
    
    def __init__(self, partitions: dict):
        """
        partitions: {
            "docs": vectorstore_docs,
            "blog": vectorstore_blog,
            "support": vectorstore_support,
            "general": vectorstore_general
        }
        """
        self.partitions = partitions
    
    def query(self, question: str, k: int = 4) -> dict:
        """Query relevant partitions and combine results."""
        
        # Determine relevant partitions
        partitions_to_query = self._select_partitions(question)
        
        # Query each partition
        all_results = []
        for partition_name in partitions_to_query:
            if partition_name not in self.partitions:
                continue
            results = self.partitions[partition_name].similarity_search(
                question, k=k
            )
            all_results.extend(results)
        
        # Remove duplicates and re-rank
        return self._rerank(question, all_results, k)
    
    def _select_partitions(self, question: str) -> list[str]:
        """Select relevant partitions based on query content."""
        
        question_lower = question.lower()
        
        # Rule-based selection
        if any(word in question_lower for word in ["tutorial", "guide", "how to", "setup"]):
            return ["docs", "general"]
        elif any(word in question_lower for word in ["release", "update", "announcement", "news"]):
            return ["blog", "general"]
        elif any(word in question_lower for word in ["error", "issue", "troubleshoot", "problem"]):
            return ["support", "general"]
        
        # Default: search all partitions
        return list(self.partitions.keys())
    
    def _rerank(self, question: str, results: list, k: int) -> dict:
        """Re-rank and deduplicate across partitions."""
        
        # Deduplicate by content
        seen = set()
        unique_results = []
        for doc in results:
            content_key = doc.page_content[:200]
            if content_key not in seen:
                seen.add(content_key)
                unique_results.append(doc)
        
        # Simple relevance scoring (in production, use cross-encoder)
        scored = []
        for doc in unique_results[:k*3]:  # Consider top k*3
            # Count keyword matches as simple score
            score = sum(1 for word in question.lower().split() 
                       if word in doc.page_content.lower())
            scored.append((doc, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return {"results": [doc for doc, _ in scored[:k]], "total": len(scored)}
```

## 4. Async Processing

**Concept**: Synchronous processing handles one query at a time, waiting for each step (retrieval, generation) to complete before starting the next. Async processing allows multiple queries to be in-flight simultaneously, maximizing throughput when queries involve I/O waits (like LLM API calls).

**Why it helps**:
- Higher throughput for I/O-bound workloads
- Better resource utilization (CPU idle while waiting for LLM)
- Non-blocking batch processing
- Responsive user experience for multiple concurrent requests

**Synchronous vs Async**:

```
Synchronous (1 query at a time):
Query1: [---retrieval---][-LLM-][-response-]
Query2:                          [---retrieval---][-LLM-][-response-]
Query3:                                                       [---retrieval---][-LLM-]

Total time: ~3x single query time

Async (parallel processing):
Query1: [---retrieval---][-LLM-][-response-]
Query2: [---retrieval---][-LLM-][-response-]
Query3: [---retrieval---][-LLM-][-response-]

Total time: ~1x single query time (if resources available)
```

**When to use**:
- High volume of concurrent requests
- LLM API calls with variable latency (100ms - 5s)
- Batch processing requirements
- Web applications with multiple simultaneous users

**Example**:
- Synchronous: 100 queries × 500ms each = 50 seconds total
- Async (10 workers): 100 queries × 500ms each = 5 seconds total (10x improvement)

```python
"""
Async RAG for High Throughput
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAG:
    """Process queries asynchronously for higher throughput."""
    
    def __init__(self, rag, max_workers: int = 10):
        self.rag = rag
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)  # Limit concurrency
    
    async def query(self, question: str) -> dict:
        """Async query with concurrency control."""
        
        async with self.semaphore:  # Limit concurrent queries
            loop = asyncio.get_running_loop()
            
            # Run sync RAG query in thread pool to avoid blocking
            result = await loop.run_in_executor(
                self.executor,
                self.rag.query,
                question
            )
            
            return result
    
    async def batch_query(self, questions: list) -> list:
        """Process multiple queries concurrently."""
        
        # Create tasks for all questions
        tasks = [self.query(q) for q in questions]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        return [
            result if not isinstance(result, Exception) 
            else {"error": str(result)}
            for result in results
        ]
    
    def query_sync(self, question: str) -> dict:
        """Synchronous interface for backwards compatibility."""
        
        return asyncio.run(self.query(question))
    
    async def batch_query_with_progress(
        self, 
        questions: list, 
        progress_callback=None
    ) -> list:
        """Process queries sequentially with progress reporting.
        
        Use batch_query() for parallel processing.
        This method is for when you need to track progress per query.
        """
        
        results = []
        total = len(questions)
        
        for i, question in enumerate(questions):
            result = await self.query(question)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
```

## 5. Data Update Strategies

**Concept**: Production RAG systems need to handle evolving data. Rather than rebuilding the entire index when documents change, incremental updates modify only the affected portions, minimizing downtime and compute costs.

**Why it helps**:
- Avoid expensive full index rebuilds
- Keep search results current with data changes
- Reduce compute costs for updates
- Enable real-time or near-real-time data freshness

**Update Patterns**:

| Pattern | Frequency | Use Case |
|---------|-----------|----------|
| **Incremental** | Real-time | New documents, corrections |
| **Batch scheduled** | Hourly/daily | Large data imports |
| **Delta updates** | Event-driven | Webhooks from data sources |
| **Full rebuild** | Weekly/monthly | Schema changes, quality refresh |

**When to use**:
- Documents change frequently (news, updates, revisions)
- Need near-real-time search freshness
- Large document collections (can't rebuild daily)
- Continuous data ingestion pipelines

**Example Update Flow**:
```
New Document arrives
       │
       ▼
┌──────────────────┐
│  Detect Change   │ ─── Is this new? ────► Add to index
└────────┬─────────┘
         │Is this changed?
         ▼
┌──────────────────┐
│  Update in Index │ ─── Delete old ──► Insert new
└────────┬─────────┘
         │Is this deleted?
         ▼
┌──────────────────┐
│ Delete from Index│
└──────────────────┘
```

```python
"""
Incremental Index Updates
"""

from typing import Callable

class IncrementalIndexer:
    """Update index incrementally without full rebuilds."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def add_documents(self, documents: list, batch_size: int = 100):
        """Add documents in batches."""
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Add to vector store
            self.vectorstore.add_documents(batch)
            
            current_batch = i // batch_size + 1
            print(f"Indexed batch {current_batch}/{total_batches}")
    
    def delete_documents(self, doc_ids: list):
        """Delete documents by ID."""
        
        if doc_ids:
            self.vectorstore.delete(doc_ids)
            print(f"Deleted {len(doc_ids)} documents")
    
    def update_documents(self, updates: dict):
        """
        Update existing documents atomically.
        
        updates: {doc_id: new_document}
        """
        
        for doc_id, new_doc in updates.items():
            # Delete old version
            self.vectorstore.delete([doc_id])
            
            # Add new version
            self.vectorstore.add_documents([new_doc])
        
        print(f"Updated {len(updates)} documents")
    
    def incremental_sync(
        self, 
        new_documents: list, 
        get_doc_id: Callable = None
    ):
        """
        Synchronize index with source of truth.
        
        Detects additions, updates, and deletions by comparing
        with existing documents.
        
        Args:
            new_documents: List of current documents from source
            get_doc_id: Function to extract document ID from document
        """
        
        get_id = get_doc_id or (lambda doc: doc.metadata.get("id"))
        
        # Get existing IDs
        existing_data = self.vectorstore.get()
        existing_ids = set(existing_data.get("ids", []))
        
        # Get new IDs
        new_ids = set()
        for doc in new_documents:
            doc_id = get_id(doc)
            if doc_id:
                new_ids.add(doc_id)
        
        # Determine changes
        to_add = new_ids - existing_ids
        to_delete = existing_ids - new_ids
        to_update = existing_ids & new_ids
        
        # Apply changes
        if to_add:
            docs_to_add = [
                doc for doc in new_documents 
                if get_id(doc) in to_add
            ]
            self.add_documents(docs_to_add)
        
        if to_delete:
            self.delete_documents(list(to_delete))
        
        # For updates, compare content hashes or use change detection
        print(f"Synced: {len(to_add)} added, {len(to_delete)} deleted, "
              f"{len(to_update)} checked for updates")
```

## 6. Monitoring & Observability

**Concept**: To scale effectively, you need visibility into system performance. Monitoring tracks key metrics (latency, throughput, errors) while observability provides deeper insight into why issues occur. Without this data, scaling decisions are guesswork.

**Key Metrics to Track**:

| Category | Metric | Target | Why It Matters |
|----------|--------|--------|----------------|
| **Latency** | P50 latency | < 200ms | User experience baseline |
| **Latency** | P95 latency | < 500ms | Most user requests |
| **Latency** | P99 latency | < 1s | Edge cases matter |
| **Throughput** | Queries/second | Depends on scale | Capacity planning |
| **Cache** | Hit rate | > 60% | Cost optimization |
| **Errors** | Error rate | < 0.1% | System reliability |
| **Retrieval** | Avg docs retrieved | 3-5 | Relevance tuning |
| **LLM** | Token usage | Track trends | Cost monitoring |

**Why it helps**:
- Identifies bottlenecks before they become outages
- Informs scaling decisions with data
- Tracks cost efficiency
- Enables SLA compliance
- Helps debug production issues

**When to use**:
- Always (this is essential for production)
- Before and after scaling changes
- To validate cache effectiveness
- To set performance budgets

**Example Dashboard Metrics**:
```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Metrics Dashboard                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Latency Distribution          Cache Performance            │
│  ┌──────────────────┐        ┌──────────────────┐           │
│  │ P50: 150ms       │        │ Hits: 85%        │           │
│  │ P95: 420ms       │        │ Misses: 15%      │           │
│  │ P99: 890ms       │        │ Saved: $45/hr    │           │
│  └──────────────────┘        └──────────────────┘           │
│                                                             │
│  Query Throughput           Error Rate                      │
│  ┌──────────────────┐        ┌──────────────────┐           │
│  │ Current: 150/s   │        │ Errors: 0.05%    │           │
│  │ Peak: 300/s      │        │ Healthy: ✓       │           │
│  └──────────────────┘        └──────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
"""
RAG Monitoring
"""

import time
from dataclasses import dataclass, field
from typing import List
from collections import deque

@dataclass
class RAGMetrics:
    """Track RAG performance metrics with rolling averages."""
    
    # Counters
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    
    # Latency tracking (rolling window)
    latency_window: int = 1000  # Keep last 1000 measurements
    retrieval_latencies: deque = field(default_factory=deque)
    generation_latencies: deque = field(default_factory=deque)
    total_latencies: deque = field(default_factory=deque)
    
    # Token tracking
    total_tokens: int = 0
    
    def record_query(
        self, 
        latency_ms: float, 
        cached: bool,
        retrieval_ms: float = 0,
        generation_ms: float = 0,
        tokens_used: int = 0
    ):
        """Record a query's metrics."""
        
        self.total_queries += 1
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Track rolling latencies
        self.retrieval_latencies.append(retrieval_ms)
        self.generation_latencies.append(generation_ms)
        self.total_latencies.append(latency_ms)
        
        # Trim window
        if len(self.retrieval_latencies) > self.latency_window:
            self.retrieval_latencies.popleft()
        if len(self.generation_latencies) > self.latency_window:
            self.generation_latencies.popleft()
        if len(self.total_latencies) > self.latency_window:
            self.total_latencies.popleft()
        
        # Token tracking
        self.total_tokens += tokens_used
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    def _percentile(self, values: deque, percentile: float) -> float:
        """Calculate percentile from deque."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        
        total = max(1, self.total_queries)
        cache_total = self.cache_hits + self.cache_misses
        
        return {
            # Query counts
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            
            # Cache performance
            "cache_hit_rate": self.cache_hits / max(1, cache_total),
            
            # Latency percentiles (ms)
            "latency_p50": self._percentile(self.total_latencies, 50),
            "latency_p95": self._percentile(self.total_latencies, 95),
            "latency_p99": self._percentile(self.total_latencies, 99),
            "avg_retrieval_ms": sum(self.retrieval_latencies) / max(1, len(self.retrieval_latencies)),
            "avg_generation_ms": sum(self.generation_latencies) / max(1, len(self.generation_latencies)),
            
            # Error tracking
            "error_count": self.error_count,
            "error_rate": self.error_count / total,
            
            # Token usage
            "total_tokens": self.total_tokens,
        }


# Usage
metrics = RAGMetrics()

class MonitoredRAG:
    """RAG with comprehensive monitoring."""
    
    def __init__(self, rag):
        self.rag = rag
        self.metrics = metrics
    
    def query(self, question: str, use_cache: bool = True) -> dict:
        """Query with automatic metric recording."""
        
        overall_start = time.time()
        
        try:
            # Retrieval timing
            t0 = time.time()
            retrieval_results = self.rag.retrieve(question)
            t_retrieval = (time.time() - t0) * 1000
            
            # Check if cached (if rag supports it)
            cached = getattr(retrieval_results, 'cached', False)
            
            # Generation timing
            t1 = time.time()
            result = self.rag.generate(question, retrieval_results)
            t_generation = (time.time() - t1) * 1000
            
            total_ms = (time.time() - overall_start) * 1000
            
            # Estimate tokens (in production, track from LLM response)
            tokens = len(question.split()) * 2  # Rough estimate
            
            self.metrics.record_query(
                total_ms, cached, t_retrieval, t_generation, tokens
            )
            
            return result
            
        except Exception as e:
            self.metrics.record_error()
            raise
```

## 7. Vector Database Scaling

**Concept**: As your vector database grows beyond 10M+ vectors, you need specialized strategies to maintain performance. This includes sharding (distributing data across multiple databases), replication (copying data for read scaling), and index optimization (tuning the ANN algorithm parameters).

**Why it helps**:
- Maintains sub-100ms query latency at scale
- Enables horizontal scaling of storage
- Provides high availability through replication
- Reduces cost by distributing load

**Scaling Strategies**:

| Strategy | Scale | Tradeoff |
|----------|-------|----------|
| **Single large index** | Up to 10M vectors | Simple, limited scaling |
| **Namespace partitioning** | Up to 100M vectors | Requires routing logic |
| **Sharding** | 100M+ vectors | Complexity, cross-shard queries |
| **Read replicas** | High read throughput | Stale reads possible |
| **ANN tuning** | Any scale | Memory vs accuracy |

**When to use**:
- Index exceeds 10M vectors
- Query latency increasing despite optimization
- Need high availability (multi-AZ deployment)
- Read throughput exceeds single node capacity

**Example - Sharding by namespace**:
```
                    User Query
                        │
                        ▼
              ┌─────────────────┐
              │   Router /      │
              │   Meta-index    │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Shard 1 │   │ Shard 2 │   │ Shard 3 │
   │ docs/   │   │ blog/   │   │ support/│
   │ 5M vec  │   │ 3M vec  │   │ 2M vec  │
   └─────────┘   └─────────┘   └─────────┘
```

```python
"""
Vector Database Sharding and Scaling
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ShardConfig:
    """Configuration for a vector database shard."""
    name: str
    namespace: str
    vectorstore: any
    vector_count: int = 0

class ShardedVectorStore:
    """Distribute vectors across multiple shards."""
    
    def __init__(self):
        self.shards: Dict[str, ShardConfig] = {}
        self.meta_index = None
    
    def add_shard(self, name: str, namespace: str, vectorstore):
        """Add a new shard."""
        self.shards[name] = ShardConfig(
            name=name,
            namespace=namespace,
            vectorstore=vectorstore
        )
    
    def _get_relevant_shards(self, query: str) -> List[str]:
        """Determine which shards to query based on query content."""
        
        query_lower = query.lower()
        
        # Simple keyword-based routing
        if any(word in query_lower for word in ["tutorial", "guide", "reference"]):
            return ["docs"]
        elif any(word in query_lower for word in ["post", "article", "blog", "news"]):
            return ["blog"]
        elif any(word in query_lower for word in ["help", "error", "issue", "support"]):
            return ["support"]
        
        # Return all shards for general queries
        return list(self.shards.keys())
    
    def add_documents(self, documents: List, namespace: str = "default"):
        """Add documents to appropriate shard."""
        
        for shard_name, shard in self.shards.items():
            if shard.namespace == namespace:
                shard.vectorstore.add_documents(documents)
                shard.vector_count += len(documents)
                return
        
        # Create new shard if namespace doesn't exist
        raise ValueError(f"Namespace {namespace} not found. Add shard first.")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        namespace: Optional[str] = None
    ) -> List:
        """Search across relevant shards."""
        
        # Determine shards to search
        if namespace:
            shard_names = [s for s, shard in self.shards.items() 
                         if shard.namespace == namespace]
        else:
            shard_names = self._get_relevant_shards(query)
        
        # Query each shard
        all_results = []
        for shard_name in shard_names:
            if shard_name not in self.shards:
                continue
            shard = self.shards[shard_name]
            results = shard.vectorstore.similarity_search(query, k=k)
            all_results.extend(results)
        
        # Deduplicate and return top k
        return self._deduplicate(all_results, k)
    
    def _deduplicate(self, results: List, k: int) -> List:
        """Remove duplicate results based on content."""
        
        seen = set()
        unique = []
        for doc in results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
                if len(unique) >= k:
                    break
        return unique
    
    def get_stats(self) -> Dict:
        """Get statistics for all shards."""
        
        total_vectors = sum(sh.vector_count for sh in self.shards.values())
        return {
            "total_vectors": total_vectors,
            "num_shards": len(self.shards),
            "shards": {
                name: {"namespace": sh.namespace, "vectors": sh.vector_count}
                for name, sh in self.shards.items()
            }
        }


class VectorDBReplica:
    """Read replica for scaling read throughput."""
    
    def __init__(self, primary, replicas: List):
        self.primary = primary
        self.replicas = replicas
        self.current_replica = 0
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        """Search using replica, fallback to primary."""
        
        if self.replicas:
            # Round-robin across replicas
            replica = self.replicas[self.current_replica]
            self.current_replica = (self.current_replica + 1) % len(self.replicas)
            try:
                return replica.similarity_search(query, k)
            except Exception:
                # Fallback to primary
                return self.primary.similarity_search(query, k)
        
        return self.primary.similarity_search(query, k)
    
    def add_documents(self, documents: List):
        """Write to primary only."""
        self.primary.add_documents(documents)
```

---

## 8. LLM Scaling

**Concept**: LLM calls are often the slowest and most expensive part of RAG. Scaling LLM components involves batching requests, managing rate limits, implementing fallbacks to smaller/faster models, and optimizing token usage.

**Why it helps**:
- Reduces LLM latency through batching
- Manages API rate limits gracefully
- Reduces costs with smaller/faster models when appropriate
- Improves reliability with fallback chains

**Scaling Strategies**:

| Strategy | Latency Impact | Cost Impact | Use Case |
|----------|----------------|-------------|----------|
| **Token batching** | -30-50% | -40-60% | High volume similar queries |
| **Model fallbacks** | Variable | -50-80% | Cost-sensitive apps |
| **Streaming** | Perceived faster | Same | User-facing apps |
| **Caching responses** | < 1ms | -80%+ | Repeated queries |
| **Smaller models** | -60-80% | -90%+ | Simple queries |

**When to use**:
- LLM latency > 50% of total response time
- API rate limits being hit
- Need to reduce LLM costs
- Variable query complexity (simple vs complex)

**Example - Tiered model fallback**:
```
Query Complexity Detection
        │
        ▼
┌─────────────────┐
│ Simple query?   │──Yes──► Use fast model (llama3.2)
│ (single topic)  │         Latency: ~200ms, Cost: $0.001
└────────┬────────┘
         │No
         ▼
┌─────────────────┐
│ Medium query?   │──Yes──► Use standard model (gpt-4o-mini)
│ (multi-part)    │         Latency: ~1s, Cost: $0.01
└────────┬────────┘
         │No
         ▼
┌─────────────────┐
│ Complex query?  │───────► Use powerful model (gpt-4o)
│ (reasoning)     │          Latency: ~5s, Cost: $0.10
└─────────────────┘
```

```python
"""
LLM Scaling with Batching and Fallbacks
"""

import asyncio
import time
from typing import List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class ModelTier(Enum):
    """LLM model tiers for cost-latency optimization."""
    FAST = "fast"           # Local/fast model
    STANDARD = "standard"   # Standard API model
    POWERFUL = "powerful"  # Most capable model

@dataclass
class LLMConfig:
    """Configuration for an LLM tier."""
    name: str
    model: any
    max_tokens: int
    latency_ms: float  # Typical latency
    cost_per_1k_tokens: float

class TieredLLM:
    """LLM with tiered model selection and batching."""
    
    def __init__(self):
        self.tiers: Dict[ModelTier, LLMConfig] = {}
        self.default_tier = ModelTier.STANDARD
    
    def register_tier(self, tier: ModelTier, config: LLMConfig):
        """Register an LLM tier."""
        self.tiers[tier] = config
    
    def _estimate_complexity(self, query: str, context: str = "") -> ModelTier:
        """Estimate query complexity to select appropriate tier."""
        
        # Simple heuristics for demonstration
        # In production, use a lightweight classifier or LLM
        
        query_lower = query.lower()
        
        # Check for complexity indicators
        complex_indicators = [
            "compare", "analyze", "evaluate", "synthesize",
            "explain why", "reason", "implications"
        ]
        multi_part_indicators = [" and ", " or ", " also ", " furthermore"]
        
        # Count indicators
        complex_count = sum(1 for word in complex_indicators if word in query_lower)
        multi_part_count = sum(1 for phrase in multi_part_indicators if phrase in query_lower)
        
        # Decision logic
        if complex_count >= 2 or len(query.split()) > 50:
            return ModelTier.POWERFUL
        elif complex_count >= 1 or multi_part_count >= 1 or len(context) > 2000:
            return ModelTier.STANDARD
        else:
            return ModelTier.FAST
    
    async def generate(
        self, 
        prompt: str, 
        tier: Optional[ModelTier] = None,
        force_tier: bool = False
    ) -> str:
        """Generate response using appropriate tier."""
        
        # Auto-select tier if not forced
        if tier is None:
            tier = self._estimate_complexity(prompt)
        
        config = self.tiers.get(tier, self.tiers[self.default_tier])
        
        try:
            response = await config.model.agenerate([prompt])
            return response.generations[0].text
        except Exception:
            # Fallback to standard tier
            if tier != ModelTier.STANDARD and not force_tier:
                config = self.tiers[ModelTier.STANDARD]
                response = await config.model.agenerate([prompt])
                return response.generations[0].text
            raise


class LLMBatcher:
    """Batch multiple LLM requests for efficiency."""
    
    def __init__(self, llm, batch_size: int = 10, max_wait_ms: int = 100):
        self.llm = llm
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: List[asyncio.Future] = []
    
    async def generate(self, prompt: str) -> str:
        """Add to batch and wait for result."""
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        self.pending.append(future)
        
        # Process batch if full or wait for timeout
        if len(self.pending) >= self.batch_size:
            await self._process_batch()
        else:
            # Schedule timeout-based processing
            asyncio.create_task(self._delayed_process())
        
        return await future
    
    async def _delayed_process(self):
        """Process batch after max wait time."""
        
        await asyncio.sleep(self.max_wait_ms / 1000)
        if self.pending:
            await self._process_batch()
    
    async def _process_batch(self):
        """Process all pending requests as a batch."""
        
        if not self.pending:
            return
        
        # Get all prompts from pending futures
        # In production, would batch API calls
        # This is simplified for demonstration
        batch = self.pending[:]
        self.pending = []
        
        # Complete all futures (simplified)
        for future in batch:
            if not future.done():
                future.set_result("batch_result")
```

---

## 9. Auto-scaling

**Concept**: Auto-scaling automatically adjusts compute resources based on demand. Unlike manual scaling (adding instances when you notice problems), auto-scaling responds automatically to metrics like CPU usage, request queue depth, or custom metrics.

**Why it helps**:
- Handles traffic spikes without manual intervention
- Optimizes costs by scaling down during low traffic
- Maintains SLA during high load
- Reduces operational overhead

**Scaling Triggers**:

| Metric | Scale Up When | Scale Down When |
|--------|--------------|-----------------|
| **CPU Usage** | > 70% for 5 min | < 30% for 10 min |
| **Memory** | > 80% | < 50% |
| **Request Queue** | > 10 pending | < 2 for 15 min |
| **Custom (QPS)** | > 80% capacity | < 30% capacity |
| **Latency P95** | > 1s | < 500ms for 10 min |

**When to use**:
- Traffic patterns are unpredictable
- Need hands-off scaling
- Variable workloads (daily/weekly patterns)
- Cost optimization is important

**Example - Kubernetes HPA configuration**:
```yaml
# Kubernetes Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom metric scaling
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

```python
"""
Auto-scaling Configuration and Patterns
"""

from dataclasses import dataclass
from typing import Callable, Optional
import time

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    
    # Instance limits
    min_instances: int = 1
    max_instances: int = 10
    
    # Scale up
    scale_up_threshold: float = 0.70  # 70% CPU/memory
    scale_up_cooldown_seconds: int = 60
    scale_up_increment: int = 1
    
    # Scale down
    scale_down_threshold: float = 0.30  # 30% CPU/memory
    scale_down_cooldown_seconds: int = 300  # 5 minutes
    scale_down_decrement: int = 1

class ScalingController:
    """Simple scaling controller for demonstration."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_instances = config.min_instances
        self.last_scale_up = 0
        self.last_scale_down = 0
    
    def evaluate(
        self, 
        cpu_usage: float, 
        memory_usage: float,
        request_queue_depth: int = 0
    ) -> Optional[int]:
        """Evaluate scaling needs based on current metrics.
        
        Returns:
            New instance count if scaling needed, None otherwise.
        """
        
        now = time.time()
        
        # Check scale up conditions
        if cpu_usage > self.config.scale_up_threshold or memory_usage > self.config.scale_up_threshold:
            if request_queue_depth > 10 or cpu_usage > 0.85:
                if now - self.last_scale_up > self.config.scale_up_cooldown_seconds:
                    if self.current_instances < self.config.max_instances:
                        self.current_instances += self.config.scale_up_increment
                        self.last_scale_up = now
                        return self.current_instances
        
        # Check scale down conditions
        if cpu_usage < self.config.scale_down_threshold and memory_usage < self.config.scale_down_threshold:
            if request_queue_depth < 2:
                if now - self.last_scale_down > self.config.scale_down_cooldown_seconds:
                    if self.current_instances > self.config.min_instances:
                        self.current_instances -= self.config.scale_down_decrement
                        self.last_scale_down = now
                        return self.current_instances
        
        return None
    
    def get_stats(self) -> dict:
        """Get current scaling stats."""
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.config.min_instances,
            "max_instances": self.config.max_instances,
            "can_scale_up": self.current_instances < self.config.max_instances,
            "can_scale_down": self.current_instances > self.config.min_instances,
        }


class QueueBasedScaler:
    """Scale based on request queue depth."""
    
    def __init__(
        self,
        queue: any,  # Message queue (SQS, RabbitMQ, etc.)
        config: ScalingConfig
    ):
        self.queue = queue
        self.config = config
        self.instances = config.min_instances
    
    async def evaluate(self) -> Optional[int]:
        """Evaluate scaling based on queue depth."""
        
        queue_depth = await self.queue.get_depth()
        avg_depth_per_instance = queue_depth / max(1, self.instances)
        
        # Scale based on queue depth per instance
        if avg_depth_per_instance > 5 and self.instances < self.config.max_instances:
            self.instances = min(self.instances + 1, self.config.max_instances)
            return self.instances
        
        if avg_depth_per_instance < 1 and self.instances > self.config.min_instances:
            self.instances = max(self.instances - 1, self.config.min_instances)
            return self.instances
        
        return None
```

---

## Summary

| Pattern | When to Use | Complexity | Impact |
|---------|-------------|------------|--------|
| **Horizontal Scaling** | High concurrent users | Low | Linear capacity |
| **Caching** | Repeated queries | Low | -80% latency, -60% cost |
| **Index Partitioning** | Large indexes | Medium | 3-10x faster retrieval |
| **Async Processing** | High throughput | Medium | 5-10x throughput |
| **Data Updates** | Changing data | Medium | Real-time freshness |
| **Monitoring** | Always | Low | Visibility |
| **Vector DB Scaling** | > 10M vectors | High | Horizontal scale |
| **LLM Scaling** | Cost/latency issues | Medium | -50-80% cost |
| **Auto-scaling** | Variable traffic | Medium | Hands-off operation |

### Recommended Starting Point

1. **Always start with**: Monitoring (visibility first)
2. **Then add**: Caching (quick wins, easy to implement)
3. **As load grows**: Horizontal scaling + Async processing
4. **As data grows**: Index partitioning
5. **As costs matter**: LLM scaling + more caching
6. **For scale**: Vector DB sharding + Auto-scaling

### Common Pitfalls

- **Scaling too early**: Measure first, scale when needed
- **Ignoring caching**: Expensive before you need to scale
- **No monitoring**: Can't know if scaling helps
- **Scaling one component**: Bottlenecks move; profile first
- **Aggressive scale-down**: Leaves you vulnerable to traffic spikes

### Scaling Decision Guide

```
What's your bottleneck?
        │
        ▼
┌───────────────────┐
│ High latency?     │──► Add caching + LLM scaling
│ (> 500ms)         │
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│ High cost?        │──► LLM scaling + better caching
│ (LLM calls)       │
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│ High throughput?  │──► Horizontal scaling + async
│ (> 100 RPS)       │
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│ Large data?       │──► Index partitioning + vector scaling
│ (> 10M vectors)   │
└───────────────────┘
```

---

## References

### Official Documentation

| Resource | Description |
|----------|-------------|
| [Kubernetes](https://kubernetes.io/docs/) | K8s documentation |
| [Docker](https://docs.docker.com/) | Docker docs |
| [Pinecone Scaling](https://docs.pinecone.io/guides/indexes/pods/scale-pod-based-indexes) | Pinecone scaling |
| [Qdrant Clustering](https://qdrant.tech/documentation/manage-data/collections/) | Qdrant scaling |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [RAG at Scale](https://medium.com/@mbentaher1/scaling-rag-systems-in-production-7023a1c9aba0) | Enterprise patterns |
| [Horizontal Scaling RAG](https://callsphere.tech/blog/production-rag-architecture-caching-monitoring-scaling-pipelines) | Multi-node RAG |
| [Vector DB Scaling](https://docs.pinecone.io/guides/get-started/test-at-scale) | Vector scaling |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [kubernetes/examples](https://github.com/kubernetes/examples) | K8s examples |

---

*Previous: [Cost Optimization](cost-optimization.md)*

*Next: [Research Directions](../5-pros-cons/research-directions.md)*
