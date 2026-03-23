# Reranking for RAG

## Overview

Reranking improves retrieval quality by re-scoring and re-ordering initial retrieval results using a more sophisticated model. It's typically used as a second-stage refinement after the initial retrieval pass.

## Why Reranking Matters

```
Two-Stage Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                      Two-Stage Retrieval                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query ──► Stage 1: Initial Retrieval ──► Stage 2: Reranking ──► Final  │
│              (Fast, less precise)          (Slower, more precise)       │
│                                                                         │
│  ┌─────────────┐                         ┌─────────────┐                │
│  │ 1. Doc A    │                         │ 1. Doc B    │                │
│  │ 2. Doc B    │      ───────►           │ 2. Doc A    │  ← Reordered!  │
│  │ 3. Doc C    │                         │ 3. Doc C    │                │
│  │ 4. Doc D    │                         │ 4. Doc D    │                │
│  │ 5. Doc E    │                         │ 5. Doc E    │                │
│  └─────────────┘                         └─────────────┘                │
│                                                                         │
│  Usually: 100-1000 docs → Top 10-50 → Final: Top 3-5                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Bi-Encoder vs Cross-Encoder

Understanding the difference is key to understanding reranking:

```python
# BI-ENCODER (used in initial retrieval)
# - Encodes query and document SEPARATELY
# - Fast (can pre-compute document embeddings)
# - Less accurate (no interaction between query-doc)
query_emb = encoder.encode(query)      # [768]
doc_emb = encoder.encode(document)    # [768]
score = cosine_similarity(query_emb, doc_emb)

# CROSS-ENCODER (used in reranking)
# - Encodes query + document TOGETHER
# - Slower (must encode each pair)
# - More accurate (full interaction)
score = cross_encoder.predict([query, document])  # 0.95
```

## Reranking Approaches

### 1. Cross-Encoder Reranking

Most common approach using transformer models trained for relevance scoring. Use this for most reranking needs - it's free, fast, and provides good accuracy.

```python
"""
Cross-Encoder Reranking
"""

from sentence_transformers import CrossEncoder

# Initialize cross-encoder (lightweight, fast)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class CrossEncoderReranker:
    """Rerank using cross-encoder model."""
    
    def __init__(self, cross_encoder, base_retriever, top_k: int = 10):
        self.cross_encoder = cross_encoder
        self.base_retriever = base_retriever
        self.top_k = top_k
    
    def rerank(self, query: str, initial_k: int = 50) -> list:
        """
        Rerank documents for a query.
        
        Args:
            query: The search query
            initial_k: Number of docs to retrieve in first stage
        """
        
        # Stage 1: Initial retrieval (get more candidates)
        initial_docs = self.base_retriever.invoke(query)
        initial_docs = initial_docs[:initial_k]
        
        # Stage 2: Prepare query-doc pairs
        doc_contents = [doc.page_content for doc in initial_docs]
        pairs = [[query, doc] for doc in doc_contents]
        
        # Stage 3: Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Stage 4: Re-order by score
        doc_scores = list(zip(initial_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k reranked
        reranked = [doc for doc, score in doc_scores[:self.top_k]]
        
        return reranked
    
    def rerank_with_scores(self, query: str, initial_k: int = 50) -> list:
        """Rerank and return with scores."""
        
        initial_docs = self.base_retriever.invoke(query)[:initial_k]
        
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder.predict(pairs)
        
        doc_scores = list(zip(initial_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:self.top_k]
```

### 2. Sentence Transformer Reranking

Using sentence-transformers library for reranking.

```python
"""
Sentence Transformer Reranking
"""

from sentence_transformers import CrossEncoder

# More powerful cross-encoders available
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Even stronger models
cross_encoder_stanford = CrossEncoder('cross-encoder/stanford-np/trel-erl-001')

class SentenceTransformerReranker:
    """Rerank using sentence-transformers cross-encoder."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """Rerank documents."""
        
        pairs = [[query, doc] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by scores
        doc_with_scores = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return doc_with_scores[:top_k]
```

### 3. Learning to Rank (LTR)

Using supervised learning for reranking. Use LTR when you have labeled training data and want domain-specific reranking - it requires more setup but can outperform general models on specific tasks.

```python
"""
Learning to Rank Reranking
"""

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class LTRReranker:
    """Learning to Rank reranker using gradient boosting."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.model = GradientBoostingRegressor()
        self.is_fitted = False
    
    def _extract_features(self, query: str, document: str) -> np.ndarray:
        """Extract features for query-document pair."""
        
        # Get embeddings
        query_emb = self.embedding_model.embed_query(query)
        doc_emb = self.embedding_model.embed_query(document)
        
        # Feature 1: Cosine similarity
        cosine_sim = np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        )
        
        # Feature 2: BM25 score (simplified)
        bm25_score = self._bm25_score(query, document)
        
        # Feature 3: Query length
        query_len = len(query.split())
        
        # Feature 4: Document length
        doc_len = len(document.split())
        
        return np.array([cosine_sim, bm25_score, query_len, doc_len])
    
    def _bm25_score(self, query: str, document: str) -> float:
        """Calculate BM25 score (simplified)."""
        # Simplified - in practice use rank_bm25 library
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        
        score = sum(1 for term in query_terms if term in doc_terms)
        return score / (len(query_terms) + 1)
    
    def fit(self, training_data: list):
        """Train the LTR model.
        
        training_data: list of (query, document, relevance_label) tuples
        """
        
        X = []
        y = []
        
        for query, doc, label in training_data:
            features = self._extract_features(query, doc)
            X.append(features)
            y.append(label)
        
        self.model.fit(X, y)
        self.is_fitted = True
    
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """Rerank using trained model."""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract features for all documents
        X = np.array([
            self._extract_features(query, doc) for doc in documents
        ])
        
        # Predict scores
        scores = self.model.predict(X)
        
        # Sort by score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores[:top_k]]
```

## Complete Reranking Pipeline

```python
"""
Complete Reranking Pipeline
"""

from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

class RerankingPipeline:
    """Complete two-stage retrieval pipeline with reranking."""
    
    def __init__(
        self,
        vectorstore: Chroma,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k: int = 20,
        final_k: int = 5
    ):
        self.vectorstore = vectorstore
        self.base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": initial_k}
        )
        self.cross_encoder = CrossEncoder(reranker_model)
        self.initial_k = initial_k
        self.final_k = final_k
    
    def retrieve(self, query: str) -> list:
        """Retrieve with reranking."""
        
        # Stage 1: Initial retrieval
        initial_docs = self.base_retriever.invoke(query)
        
        if not initial_docs:
            return []
        
        # Stage 2: Prepare for reranking
        doc_contents = [doc.page_content for doc in initial_docs]
        pairs = [[query, doc] for doc in doc_contents]
        
        # Stage 3: Cross-encoder scoring
        scores = self.cross_encoder.predict(pairs)
        
        # Stage 4: Re-rank
        doc_scores = list(zip(initial_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [doc for doc, _ in doc_scores[:self.final_k]]
    
    def retrieve_with_metadata(self, query: str) -> dict:
        """Retrieve with full metadata."""
        
        initial_docs = self.base_retriever.invoke(query)
        doc_contents = [doc.page_content for doc in initial_docs]
        
        pairs = [[query, doc] for doc in doc_contents]
        scores = self.cross_encoder.predict(pairs)
        
        doc_scores = list(zip(initial_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = doc_scores[:self.final_k]
        
        return {
            "query": query,
            "results": [
                {
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in top_docs
            ]
        }
```

## When to Use What

| Approach | Best For | Latency | Accuracy | Cost |
|----------|----------|---------|----------|------|
| **Cross-Encoder (MS MARCO)** | General purpose | Medium | High | Free |
| **Sentence-Transformer** | Better accuracy | Medium | Higher | Free |
| **LTR** | Domain-specific | High | Variable | Free |

## Popular Cross-Encoder Models

| Model | Description | Speed |
|-------|-------------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, good quality | Very Fast |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | More powerful | Fast |
| `cross-encoder/ms-marco-Multi-MiniLM-L-6-v2` | Multilingual | Fast |
| `cross-encoder/stanford-np/trel-erl-001` | Research model | Medium |
| `BAAI/bge-reranker-base` | BGE reranker | Medium |

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [Cross-Encoders for Information Retrieval](https://arxiv.org/abs/1904.02095) | 2019 | Cross-encoder architecture |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [Cross-Encoders - Sentence Transformers](https://sbert.net/examples/cross_encoder/training/rerankers/README.html) | Official documentation |
| [LangChain Cross Encoders](https://python.langchain.com/docs/integrations/cross_encoders/) | LangChain integration |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [The Art of RAG: Reranking](https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669) | Cross-encoder tutorial |
| [Reranking with Cohere](https://txt.cohere.com/rerank/) | Cohere reranking guide |
| [Training Reranker Models](https://huggingface.co/blog/train-reranker) | HuggingFace guide |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | SBERT library |
| [Cohere](https://github.com/cohere-ai/cohere-python) | Cohere SDK |

---

*Previous: [Query Rewriting](query-rewriting.md)*

*Next: [Sentence Window Retrieval](sentence-window-retrieval.md)*
