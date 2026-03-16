# Retrieval Systems

## Overview

The retrieval component is the heart of any RAG system. This document covers different retrieval paradigms, their implementations, and when to use each.

## Retrieval Paradigms

```
Retrieval Methods:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                      Retrieval Paradigms                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │     Dense        │  │     Sparse       │  │     Hybrid       │       │
│  │   Retrieval      │  │   Retrieval      │  │   Retrieval      │       │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤       │
│  │ • Embedding      │  │ • BM25           │  │ • Combine        │       │
│  │   similarity     │  │ • TF-IDF         │  │   Dense + Sparse │       │
│  │ • Neural         │  │ • Bag of Words   │  │ • Reciprocal     │       │
│  │   search         │  │                  │  │   Rank Fusion    │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │   Knowledge      │  │   Adaptive       │  │   Multi-Stage    │       │
│  │   Graph          │  │   Retrieval      │  │   Retrieval      │       │
│  │   Retrieval      │  │                  │  │                  │       │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤       │
│  │ • Graph          │  │ • Query          │  │ • Retrieve       │       │
│  │   traversal      │  │   analysis       │  │ • Re-rank        │       │
│  │ • Cypher         │  │ • Strategy       │  │ • Filter         │       │
│  │   queries        │  │   selection      │  │                  │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Dense Retrieval (Semantic Search)

### How It Works

```
Dense Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query "What is RAG?"                 
       │
       ▼
┌──────────────────┐
│ Embedding Model  │  ──► [0.12, -0.34, 0.78, ...]
└──────────────────┘
       │
       ▼
┌──────────────────┐      ┌──────────────────┐
│ Vector Database  │ ◄─── │ Cosine           │
│ (Index)          │      │ Similarity       │
└──────────────────┘      └──────────────────┘
       │
       ▼
┌──────────────────┐
│ Top-K Results    │
│ 1. RAG is...     │
│ 2. RAG combines  │  
│ 3. RAG helps...  │
└──────────────────┘
```

### Implementation

```python
"""
Dense Retrieval with LangChain + Ollama
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Create vector store (using Ollama embeddings - free, local)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

# Basic similarity search
results = vectorstore.similarity_search(
    query="What is RAG?",
    k=4
)

# With similarity scores
results_with_scores = vectorstore.similarity_search_with_score(
    query="What is RAG?",
    k=4
)

# Filtered search
results = vectorstore.similarity_search(
    query="What is RAG?",
    k=4,
    filter={"source": "doc1"}
)

# MMR (Max Marginal Relevance) - diverse results
results = vectorstore.max_marginal_relevance_search(
    query="What is RAG?",
    k=4,
    fetch_k=20,  # Fetch more, then select diverse
    lambda_mult=0.5  # Balance relevance vs diversity
)
```

## 2. Sparse Retrieval (BM25)

### How It Works

```
BM25 Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query "RAG system"                 
       │
       ▼
┌──────────────────┐
│ Tokenize Query   │  ──► ["RAG", "system"]
└──────────────────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│ Calculate BM25   │ ◄───│ TF × IDF         │
│ scores           │     │ scoring          │
└──────────────────┘     └──────────────────┘
       │
       ▼
┌──────────────────┐
│ Ranked Results   │
│ 1. RAG system    │
│ 2. RAG for...    │  
│ 3. System RAG    │
└──────────────────┘
```

### Implementation

```python
"""
BM25 Retrieval with LangChain
"""

from langchain_community.retrievers import BM25Retriever

# Create BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)

# Configure parameters
bm25_retriever.k1 = 1.5  # Term frequency saturation
bm25_retriever.b = 0.75  # Document length normalization

# Retrieve
results = bm25_retriever.invoke("What is RAG?")

# With custom parameters
results = bm25_retriever.invoke("What is RAG?")
```

### BM25 vs Dense

| Aspect | BM25 | Dense |
|--------|------|-------|
| **Speed** | Fast | Medium |
| **Semantic** | No (keyword) | Yes |
| **Out-of-vocab** | Poor | Good |
| **Scalability** | Excellent | Good |
| **Tuneability** | k1, b params | Model selection |

## 3. Hybrid Retrieval

### How It Works

```
Hybrid Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query "What is RAG?"                 
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│ BM25 Retriever   │     │ Dense Retriever  │
│ (keyword match)  │     │ (semantic match) │
└────────┬─────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │  Combine & Re-rank  │
         │  (Weighted/RRF)     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Top-K Results     │
         └─────────────────────┘
```

### Implementation

```python
"""
Hybrid Retrieval: Combining BM25 + Dense
"""

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# Create retrievers
bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = Chroma.from_documents(documents, embeddings).as_retriever()

# Ensemble with weighted combination
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # BM25 weight, Dense weight
)

# Retrieve
results = ensemble_retriever.invoke("What is RAG?")

# Alternative: Reciprocal Rank Fusion
def reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    """Combine multiple result lists using RRF."""
    
    scores = {}
    
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:50]  # Dedup key
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    
    # Sort by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in sorted_docs]
```

### Custom Hybrid with Re-ranking

```python
"""
Three-Stage Retrieval: Initial → Re-rank → Final
"""

from sentence_transformers import CrossEncoder

class ThreeStageRetriever:
    """Retrieve, rerank, then filter."""
    
    def __init__(self, documents, embeddings):
        # Stage 1: Dense retrieval (get more candidates)
        self.vectorstore = Chroma.from_documents(documents, embeddings)
        
        # Stage 2: Cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Stage 3: Final filter
        self.filter_fn = lambda doc: True  # Custom filter
    
    def retrieve(self, query: str, k: int = 4) -> list:
        # Stage 1: Get candidates (more than final k)
        candidates = self.vectorstore.similarity_search(query, k=k*5)
        
        # Stage 2: Re-rank
        scored = self._rerank(query, candidates)
        
        # Stage 3: Filter
        filtered = [doc for doc in scored if self.filter_fn(doc)]
        
        return filtered[:k]
    
    def _rerank(self, query: str, candidates: list) -> list:
        """Re-rank using cross-encoder."""
        
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.invoke(pairs)
        
        # Sort by score
        return [doc for doc, _ in sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )]
```

## 4. Knowledge Graph Retrieval

### How It Works

```
Knowledge Graph Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query "Who supplies Company X?"                 
       │
       ▼
┌──────────────────────────┐
│  Entity Extraction       │  ──► ["Company X", "supplier"]
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Graph Traversal (Cypher)│
│  MATCH (e:Entity)-[r]-() │  ──► Traverse relationships
│  WHERE e.name = "X"      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Return connected        │
│  entities & paths        │
└──────────────────────────┘
```

### Implementation

```python
"""
Knowledge Graph Retrieval
"""

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument

class KGRetriever:
    """Retrieve from knowledge graph."""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def retrieve_by_entities(self, query: str, entities: list) -> list:
        """Retrieve paths involving entities."""
        
        results = []
        
        for entity in entities:
            # Traverse up to 2 hops
            cypher = f"""
            MATCH path = (e:Entity)-[r*1..2]-(related)
            WHERE e.name CONTAINS '{entity}'
            RETURN path, e, r, related
            LIMIT 10
            """
            
            results.extend(self.graph.query(cypher))
        
        return results
    
    def retrieve_by_relationship(self, source: str, relation: str) -> list:
        """Find specific relationships."""
        
        cypher = f"""
        MATCH (a)-[r:{relation}]-(b)
        WHERE a.name CONTAINS '{source}'
        RETURN a, r, b
        """
        
        return self.graph.query(cypher)
```

## 5. Adaptive Retrieval

### How It Works

```
Adaptive Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query "What is RAG?"                 
       │
       ▼
┌──────────────────────────┐
│  Query Analysis (LLM)    │  ──► Determine intent
│  "semantic search"       │
└────────────┬─────────────┘
             │
             ▼
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌──────────┐    ┌──────────┐
│  Dense   │    │   BM25   │
│  Search  │    │  Search  │
└────┬─────┘    └────┬─────┘
     │                │
     └────────┬───────┘
              │
              ▼
    ┌──────────────────┐
    │  Combine Results │
    └──────────────────┘
```

### Implementation

```python
"""
Adaptive Retrieval: Choose strategy based on query
"""

class AdaptiveRetriever:
    """Select retrieval strategy per query."""
    
    def __init__(self, llm, retrievers: dict):
        self.llm = llm
        self.retrievers = retrievers
    
    def determine_strategy(self, query: str) -> str:
        """Analyze query and select strategy."""
        
        prompt = f"""Analyze this query and choose the best retrieval strategy.

Query: {query}

Strategies:
- "semantic": For conceptual, meaning-based searches
- "keyword": For specific terms, names, codes
- "hybrid": For complex queries needing both
- "kg": For relationship-based queries

Choose one:"""

        response = self.llm.invoke(prompt).content.strip().lower()
        
        # Map to available retrievers
        if "keyword" in response:
            return "bm25"
        elif "kg" in response or "relationship" in response:
            return "kg"
        elif "hybrid" in response:
            return "hybrid"
        return "semantic"
    
    def retrieve(self, query: str, k: int = 4) -> list:
        """Execute with appropriate strategy."""
        
        strategy = self.determine_strategy(query)
        
        retriever = self.retrievers.get(strategy, self.retrievers["semantic"])
        
        return retriever.invoke(query)[:k]
```

## Retrieval Evaluation

```python
"""
Evaluate Retrieval Quality
"""

def evaluate_retrieval(retriever, test_cases: list) -> dict:
    """Evaluate retrieval on test cases."""
    
    results = {
        "precision_at_k": [],
        "recall_at_k": [],
        "mrr": []  # Mean Reciprocal Rank
    }
    
    for query, relevant in test_cases:
        retrieved = retriever.invoke(query)
        
        # Calculate metrics
        retrieved_ids = [doc.id for doc in retrieved]
        
        # Precision@K
        k = len(relevant)
        precision = len(set(retrieved_ids[:k]) & set(relevant)) / k
        results["precision_at_k"].append(precision)
        
        # Recall@K
        recall = len(set(retrieved_ids[:k]) & set(relevant)) / len(relevant)
        results["recall_at_k"].append(recall)
        
        # MRR
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                results["mrr"].append(1/i)
                break
    
    # Average
    return {k: sum(v)/len(v) for k, v in results.items()}
```

## Summary: When to Use What

| Scenario | Recommended Retrieval |
|----------|---------------------|
| General Q&A | Dense (semantic) |
| Specific terms/codes | BM25 or Hybrid |
| Complex relationships | Knowledge Graph |
| Variable queries | Adaptive |
| High precision needed | Three-stage (rerank) |
| Large scale | Hybrid with caching |

---

## References

### Retrieval Techniques

| Resource | Description |
|----------|-------------|
| [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25) | Wikipedia explanation of BM25 |
| [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Original RRF paper |

### Reranking

| Resource | Description |
|----------|-------------|
| [Cross-Encoders - Sentence Transformers](https://sbert.net/examples/cross_encoder/training/rerankers/README.html) | Official documentation |
| [Reranking with Cross-Encoders](https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669) | Tutorial on cross-encoder reranking |
| [Cohere Rerank](https://cohere.com/rerank) | Commercial reranking API |
| [Training Reranker Models](https://huggingface.co/blog/train-reranker) | HuggingFace guide |

### Hybrid Search

| Resource | Description |
|----------|-------------|
| [Advanced RAG Techniques - Weaviate](https://weaviate.io/blog/advanced-rag) | Comprehensive guide to hybrid search and more |
| [Neo4j Advanced RAG](https://neo4j.com/blog/genai/advanced-rag-techniques/) | 15 advanced RAG techniques |

---

*Next: [Evaluation Metrics](evaluation-metrics.md)*
