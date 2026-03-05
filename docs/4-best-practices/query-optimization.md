# Query Optimization

## Overview

Optimizing queries is crucial for RAG performance. This includes query understanding, rewriting, and retrieval strategy selection.

## Query Optimization Pipeline

```
Query Optimization Flow:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Query
     │
     ▼
┌─────────────┐
│   Query     │
│  Preprocess │ ──► Lowercase, trim, normalize
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Query     │─────── Expansion / Rewriting
│   rewriting │       (Make query better)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Strategy  │─────── Select: similarity, MMR, hybrid
│   Selection │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Retrieval │
│   + Ranking │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Response   │
│  Generation │
└─────────────┘
```

## 1. Query Preprocessing

```python
"""
Query Preprocessing
"""

import re
from typing import List

class QueryPreprocessor:
    """Clean and normalize queries."""
    
    def preprocess(self, query: str) -> str:
        """Clean query."""
        
        # Lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters (keep important punctuation)
        query = re.sub(r'[^\w\s.,!?-]', '', query)
        
        return query
    
    def expand(self, query: str) -> List[str]:
        """Generate query variations."""
        
        # Add common variations
        expansions = [
            query,
            query + "?",  # Question form
            "What is " + query,  # Question prefix
            query.replace("?", ""),  # Without question mark
        ]
        
        return expansions

# Usage
preprocessor = QueryPreprocessor()
cleaned = preprocessor.preprocess("  What is RAG???  ")
print(cleaned)  # "what is rag?"
```

## 2. Query Rewriting

```python
"""
Query Rewriting with LLM
"""

from langchain_ollama import ChatOllama

class QueryRewriter:
    """Rewrite queries for better retrieval."""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3.2")
    
    def rewrite(self, query: str) -> str:
        """Rewrite query to improve retrieval."""
        
        prompt = f"""Rewrite this query to improve search results.
        
Original: {query}

Guidelines:
- Expand abbreviations
- Add relevant synonyms
- Clarify ambiguous terms
- Make more specific if too vague

Rewritten query:"""

        rewritten = self.llm.invoke(prompt)
        return rewritten.strip()
    
    def expand_for_search(self, query: str) -> str:
        """Generate multiple query variations."""
        
        prompt = f"""Generate 3 different versions of this query for parallel search.
        
Query: {query}

Format as comma-separated list:"""

        response = self.llm.invoke(prompt)
        variations = [query] + [v.strip() for v in response.split(",")]
        
        return variations
    
    def decompose(self, query: str) -> List[str]:
        """Break complex query into sub-queries."""
        
        prompt = f"""Break this complex question into simpler sub-questions.

Question: {query}

Format as numbered list:"""

        response = self.llm.invoke(prompt)
        # Parse numbered list
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        sub_queries = [l.split(".", 1)[-1].strip() for l in lines if l[0].isdigit()]
        
        return sub_queries if sub_queries else [query]
```

## 3. HyDE (Hypothetical Document Embeddings)

```python
"""
HyDE: Generate hypothetical documents
"""

class HyDERetriever:
    """Use HyDE for better retrieval."""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def retrieve(self, query: str, k: int = 4) -> list:
        """Retrieve using HyDE."""
        
        # Step 1: Generate hypothetical answer
        prompt = f"""Write a hypothetical answer to this question.
Just write what a good answer might contain, in 2-3 sentences.

Question: {query}

Hypothetical answer:"""

        hypothetical = self.llm.invoke(prompt)
        
        # Step 2: Embed both query and hypothetical
        query_embedding = self.vectorstore.embedding.embed_query(query)
        hypo_embedding = self.vectorstore.embedding.embed_query(hypothetical)
        
        # Step 3: Use hypothetical for retrieval
        # (averaging or using hypothetical alone)
        combined_embedding = [
            (q + h) / 2 for q, h in zip(query_embedding, hypo_embedding)
        ]
        
        # Search
        results = self.vectorstore.search(
            query_embedding=combined_embedding,
            k=k
        )
        
        return results

# In LangChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Using HyDE pattern
hyde_retriever = HyDERetriever(vectorstore, llm)
```

## 4. Query Decomposition

```python
"""
Multi-Query Retrieval
"""

class MultiQueryRetriever:
    """Generate and execute multiple queries."""
    
    def __init__(self, vectorstore, llm, original_retriever):
        self.vectorstore = vectorstore
        self.llm = llm
        self.original = original_retriever
    
    def retrieve(self, query: str) -> list:
        """Retrieve from multiple query variations."""
        
        # Generate variations
        prompt = f"""Generate 3 different search queries for this question.
        
Question: {query}

Return as comma-separated list:"""

        response = self.llm.invoke(prompt)
        queries = [query] + [q.strip() for q in response.split(",")]
        
        # Retrieve for each
        all_docs = []
        seen_content = set()
        
        for q in queries[:3]:  # Limit to 3
            docs = self.vectorstore.similarity_search(q, k=4)
            
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        # Re-rank by relevance
        return self._rerank(query, all_docs)
    
    def _rerank(self, query: str, docs: list) -> list:
        """Re-rank combined results."""
        # Simple re-ranking based on position
        return docs[:6]
```

## 5. Hybrid Search

```python
"""
Combining Keyword and Semantic Search
"""

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class HybridSearchRetriever:
    """Combine semantic and keyword search."""
    
    def __init__(self, documents: list):
        # Semantic (vector) retriever
        self.vectorstore = Chroma.from_documents(
            documents, 
            OpenAIEmbeddings()
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Keyword (BM25) retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
    
    def retrieve(self, query: str, weights: tuple = (0.5, 0.5)) -> list:
        """Combine both retrievers."""
        
        # Get results from both
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # Combine with weighting
        combined = self._combine_results(
            vector_docs, 
            bm25_docs, 
            weights
        )
        
        return combined[:4]
    
    def _combine_results(self, vec_docs: list, bm25_docs: list, weights: tuple) -> list:
        """Weighted combination of results."""
        
        doc_scores = {}
        
        # Score vector results
        for i, doc in enumerate(vec_docs):
            key = doc.page_content[:100]
            score = (len(vec_docs) - i) / len(vec_docs) * weights[0]
            doc_scores[key] = doc_scores.get(key, 0) + score
        
        # Score BM25 results
        for i, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            score = (len(bm25_docs) - i) / len(bm25_docs) * weights[1]
            doc_scores[key] = doc_scores.get(key, 0) + score
        
        # Sort and return
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Get full documents
        all_docs = vec_docs + bm25_docs
        result = []
        for key in sorted_keys:
            for doc in all_docs:
                if doc.page_content[:100] == key and doc not in result:
                    result.append(doc)
                    break
        
        return result
```

## 6. Re-ranking

```python
"""
Re-ranking Retrieved Results
"""

from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import CrossEncoder

class RerankingRetriever:
    """Re-rank results for quality."""
    
    def __init__(self, vectorstore, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vectorstore = vectorstore
        self.encoder = CrossEncoder(model_name)
    
    def retrieve(self, query: str, k: int = 10) -> list:
        """Retrieve and re-rank."""
        
        # Initial retrieval (more candidates)
        initial_docs = self.vectorstore.similarity_search(query, k=k*2)
        
        # Re-rank
        doc_texts = [doc.page_content for doc in initial_docs]
        scores = self.encoder.invoke([(query, doc) for doc in doc_texts])
        
        # Sort by score
        scored_docs = sorted(
            zip(initial_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top k
        return [doc for doc, score in scored_docs[:k]]
```

## 7. Adaptive Retrieval

```python
"""
Adaptive Query Strategy Selection
"""

class AdaptiveRetriever:
    """Select retrieval strategy based on query."""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def determine_strategy(self, query: str) -> str:
        """Determine best retrieval strategy."""
        
        prompt = f"""Analyze this query and determine the best retrieval strategy.

Query: {query}

Strategies:
- "semantic": For conceptual, meaning-based searches
- "keyword": For specific terms, names, codes
- "hybrid": For complex queries needing both
- "multi": For questions needing multiple pieces

Respond with just the strategy name:"""

        strategy = self.llm.invoke(prompt).strip().lower()
        
        # Map to actual strategy
        strategies = {
            "semantic": self._semantic_retrieve,
            "keyword": self._keyword_retrieve,
            "hybrid": self._hybrid_retrieve,
            "multi": self._multi_retrieve,
        }
        
        return strategies.get(strategy, self._semantic_retrieve)
    
    def _semantic_retrieve(self, query: str) -> list:
        return self.vectorstore.similarity_search(query, k=4)
    
    def _keyword_retrieve(self, query: str) -> list:
        """Retrieve using keyword-based BM25 approach."""
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        
        # Create BM25 retriever from documents
        bm25_retriever = BM25Retriever.from_documents(self.vectorstore._chroma_collection.get()["documents"])
        
        # Get relevant documents
        docs = bm25_retriever.get_relevant_documents(query)
        
        return docs[:4]  # Return top 4 results
    
    def _hybrid_retrieve(self, query: str) -> list:
        """Retrieve using hybrid search (keyword + semantic)."""
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        
        # Create both retrievers
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Get BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            self.vectorstore._chroma_collection.get()["documents"]
        )
        
        # Create ensemble retriever
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Weight semantic higher
        )
        
        return ensemble.get_relevant_documents(query)
    
    def _multi_retrieve(self, query: str) -> list:
        """Retrieve using multi-query approach for comprehensive results."""
        from typing import Set
        
        # Generate multiple query variations
        prompt = f"""Generate 3 different search queries for this question.
        
Question: {query}

Format as comma-separated list:"""

        response = self.llm.invoke(prompt)
        variations = [query] + [v.strip() for v in response.content.split(",") if v.strip()]
        
        # Retrieve for each variation
        all_docs = []
        seen_content: Set[str] = set()
        
        for q in variations[:3]:  # Limit to 3 variations
            docs = self.vectorstore.similarity_search(q, k=4)
            
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        # Return top 8 combined results
        return all_docs[:8]
    
    def retrieve(self, query: str) -> list:
        """Execute adaptive retrieval."""
        strategy = self.determine_strategy(query)
        return strategy(query)
```

## Summary

| Technique | When to Use | Benefit |
|-----------|-------------|---------|
| **Preprocessing** | Always | Clean input |
| **Rewriting** | Vague queries | Better matches |
| **HyDE** | Complex questions | Improved retrieval |
| **Decomposition** | Multi-part questions | Comprehensive |
| **Hybrid Search** | Mixed queries | Best of both |
| **Re-ranking** | Quality-critical | Better results |
| **Adaptive** | Variable queries | Optimized per query |

---

*Next: [Production Deployment Guide](./production-deployment.md)*
