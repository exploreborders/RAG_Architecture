# Query Optimization

## Overview

Optimizing queries is crucial for RAG performance. This includes query understanding, rewriting, and retrieval strategy selection.

> **Key insight**: The query users ask is rarely the optimal form for retrieval. Transforming queries before searching significantly improves results.

### Why Query Optimization Matters

```
User Query          Documents in Vector DB
     │                      │
     ▼                      ▼
"how RAG works?"    "Retrieval-Augmented Generation (RAG) is..."
                     "RAG combines retrieval with generation..."
                     "To implement RAG, you need..."

Problem: User uses "works"
         Docs use "implement", "combine"
         → Low similarity despite relevance!
```

Query optimization transforms the user's question to better match how information is stored.

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

**Concept**: Before sending a query to your vector database, you should clean and normalize it. This ensures consistent matching regardless of how the user formats their question.

**Why it helps**:
- Removes noise (extra spaces, special characters)
- Normalizes case so "RAG" matches "rag"
- Prevents injection attacks by removing dangerous characters

**When to use**:
- Always (this is a foundational step)
- When users might type inconsistently
- When you want to reduce false negatives from formatting differences

**Example**:
- Input: `"  What is RAG???  "`
- Output: `"what is rag?"`

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

**Concept**: Users often ask questions in ways that don't match how information is stored in your documents. Query rewriting uses an LLM to transform the user's question into a form that better matches your document store.

**Why it helps**:
- Expands abbreviations (RAG → Retrieval-Augmented Generation)
- Adds synonyms that might appear in documents
- Clarifies ambiguous terms

**When to use**:
- User uses different terminology than your documents
- Query is vague or ambiguous
- User asks partial questions (e.g., "how does it work?" without specifying what)

**Example**:
- Original: `"how does it work?"`
- Rewritten: `"how does Retrieval-Augmented Generation work?"`

```python
"""
Query Rewriting with LLM
"""

from langchain_ollama import ChatOllama

class QueryRewriter:
    """Rewrite queries for better retrieval."""
    
    def __init__(self, llm=None):
        self._llm = llm
    
    @property
    def llm(self):
        """Lazy load LLM to avoid creating new instance on every call."""
        if self._llm is None:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(model="llama3.2")
        return self._llm
    
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

        response = self.llm.invoke(prompt)
        # Handle both string and object responses
        rewritten = response.content if hasattr(response, 'content') else str(response)
        return rewritten.strip()
    
    def expand_for_search(self, query: str) -> list[str]:
        """Generate multiple query variations."""
        
        prompt = f"""Generate 3 different versions of this query for parallel search.

Query: {query}

Format as comma-separated list:"""

        response = self.llm.invoke(prompt)
        # Handle both string and object responses
        response_text = response.content if hasattr(response, 'content') else str(response)
        variations = [query] + [v.strip() for v in response_text.split(",")]
        
        return variations
    
    def decompose(self, query: str) -> list[str]:
        """Break complex query into sub-queries."""
        
        prompt = f"""Break this complex question into simpler sub-questions.

Question: {query}

Format as numbered list:"""

        response = self.llm.invoke(prompt)
        # Handle both string and object responses
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse numbered list
        lines = [l.strip() for l in response_text.split("\n") if l.strip()]
        sub_queries = [l.split(".", 1)[-1].strip() for l in lines if l and l[0].isdigit()]
        
        return sub_queries if sub_queries else [query]
```

## 3. HyDE (Hypothetical Document Embeddings)

**Concept**: Instead of embedding the user's question directly, HyDE first generates a hypothetical "ideal" answer, then embeds that. The idea is that the hypothetical answer will be closer in the embedding space to the actual relevant documents than the original question is.

**Why it helps**:
- The hypothetical answer has similar embedding characteristics to real documents
- Works well when users ask conceptual questions that don't match document wording
- Can bridge the "lexical gap" between queries and documents

**When to use**:
- Complex, conceptual questions
- When you have high-quality documents but users ask in different ways
- When semantic search alone isn't giving good results

**Example**:
- Original query: `"what is rag?"`
- Hypothetical answer: `"RAG stands for Retrieval-Augmented Generation, which is a technique that enhances LLM responses by retrieving relevant information from a knowledge base before generating answers."`
- Then embed the hypothetical answer instead of the query

```python
"""
HyDE: Generate hypothetical documents
"""

class HyDERetriever:
    """Use HyDE for better retrieval."""
    
    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
    
    def retrieve(self, query: str, k: int = 4) -> list:
        """Retrieve using HyDE."""
        
        # Step 1: Generate hypothetical answer
        prompt = f"""Write a hypothetical answer to this question.
Just write what a good answer might contain, in 2-3 sentences.

Question: {query}

Hypothetical answer:"""

        response = self.llm.invoke(prompt)
        hypothetical = response.content if hasattr(response, 'content') else str(response)
        
        # Step 2: Embed both query and hypothetical
        # Note: Use embed_query (deprecated but widely supported) or embed_documents
        query_embedding = self.embeddings.embed_query(query)
        hypo_embedding = self.embeddings.embed_query(hypothetical)
        
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

# Alternative: Use LangChain's built-in HyDE if available
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_community.document_compressors import LLMChainExtractor

# Using HyDE pattern
hyde_retriever = HyDERetriever(vectorstore, llm, embeddings)
```

## 4. Query Decomposition

**Concept**: Some questions contain multiple parts that need to be answered separately. Query decomposition breaks complex questions into simpler sub-questions, retrieves relevant documents for each, and combines the results.

**Why it helps**:
- A single query might need information from multiple sources
- Each sub-query can be more specific, leading to better retrieval
- Prevents the system from answering only part of a question

**When to use**:
- Multi-part questions (e.g., "What is X and how does Y work?")
- Questions with multiple conditions
- Complex research questions that need comprehensive answers

**Example**:
- Original: `"What is RAG and how does it compare to fine-tuning?"`
- Decomposed:
  1. `"What is RAG?"`
  2. `"What is LLM fine-tuning?"`
  3. `"RAG vs fine-tuning comparison"`

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
        # Handle both string and object responses
        response_text = response.content if hasattr(response, 'content') else str(response)
        queries = [query] + [q.strip() for q in response_text.split(",")]
        
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

**Concept**: Semantic (vector) search excels at finding conceptually similar content but misses exact keyword matches. Keyword search (BM25) finds exact matches but misses semantic relationships. Hybrid search combines both to get the best of both worlds.

**Why it helps**:
- Vector search finds: "car" ≈ "automobile"
- Keyword search finds: exact matches like model numbers, proper nouns
- Combining them catches both semantic and lexical matches

**When to use**:
- Queries with specific terms or proper nouns (names, codes, product IDs)
- When documents contain both technical terms and conceptual explanations
- When you need high precision and recall

**Example**:
- Query: `"What is GPT-4 model context length?"`
- Vector finds: conceptual info about context windows
- Keyword finds: exact "GPT-4" and "context length" matches
- Combined gives comprehensive results

```python
"""
Combining Keyword and Semantic Search
"""

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class HybridSearchRetriever:
    """Combine semantic and keyword search."""
    
    def __init__(self, documents: list, embeddings):
        from langchain_community.vectorstores import Chroma
        
        # Semantic (vector) retriever
        self.vectorstore = Chroma.from_documents(
            documents, 
            embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Keyword (BM25) retriever - use from_texts for raw text
        self.bm25_retriever = BM25Retriever.from_texts(
            [doc.page_content for doc in documents]
        )
    
    def retrieve(self, query: str, weights: tuple = (0.5, 0.5)) -> list:
        """Combine both retrievers."""
        
        # Get results from both
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
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

**Concept**: Initial retrieval fetches the most similar documents based on embedding distance, but similarity isn't the same as relevance. Re-ranking uses a more expensive but accurate model to score the relationship between the query and each retrieved document, then returns only the most relevant ones.

**Why it helps**:
- Vector similarity ≠ semantic relevance
- First-stage retrieval is fast but approximate
- Re-ranking uses cross-encoders which consider query-document pairs together

**When to use**:
- When initial retrieval has good recall but poor precision
- When answer quality is critical
- When you can afford the additional latency (re-ranking is slower)

**Example**:
- Query: `"how to improve RAG accuracy?"`
- Initial retrieval returns 10 docs about various RAG topics
- Re-ranker scores each: doc about "accuracy metrics" scores higher than doc about "RAG architecture"
- Top 3 most relevant returned

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
        
        # Ensure scores is a flat list (newer versions may return different format)
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif len(scores) > 0 and hasattr(scores[0], '__iter__') and not isinstance(scores[0], str):
            # Handle nested list case
            scores = [s[0] if isinstance(s, (list, tuple)) else s for s in scores]
        
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

**Concept**: Not all queries need the same retrieval strategy. Adaptive retrieval uses an LLM to analyze the query and choose the best strategy on the fly. This optimizes both quality and speed.

**Why it helps**:
- Simple questions don't need complex multi-query approaches
- Saves cost and latency by using only what's needed
- Different strategies work better for different query types

**When to use**:
- When query types vary widely (simple Q&A vs complex research)
- When you want to optimize for both quality and cost
- When you have multiple retrieval strategies available

**How it chooses**:
- `"semantic"`: Conceptual, meaning-based searches
- `"keyword"`: Specific terms, names, codes
- `"hybrid"`: Complex queries needing both
- `"multi"`: Questions needing multiple pieces of information

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

        response = self.llm.invoke(prompt)
        # Handle both string and object responses
        response_text = response.content if hasattr(response, 'content') else str(response)
        strategy = response_text.strip().lower()
        
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
        
        # Get documents from vectorstore - need to convert to texts for BM25
        docs_data = self.vectorstore._collection.get()
        texts = docs_data.get("documents", [])
        
        if not texts:
            return []
        
        # Create BM25 retriever from texts
        bm25_retriever = BM25Retriever.from_texts(texts)
        
        # Get relevant documents
        docs = bm25_retriever.invoke(query)
        
        return docs[:4]  # Return top 4 results
    
    def _hybrid_retrieve(self, query: str) -> list:
        """Retrieve using hybrid search (keyword + semantic)."""
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        
        # Create both retrievers
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Get texts from vectorstore
        docs_data = self.vectorstore._collection.get()
        texts = docs_data.get("documents", [])
        
        if not texts:
            return vector_retriever.invoke(query)
        
        # Get BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(texts)
        
        # Create ensemble retriever
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Weight semantic higher
        )
        
        return ensemble.invoke(query)
    
    def _multi_retrieve(self, query: str) -> list:
        """Retrieve using multi-query approach for comprehensive results."""
        from typing import Set
        
        # Generate multiple query variations
        prompt = f"""Generate 3 different search queries for this question.
        
Question: {query}

Format as comma-separated list:"""

        response = self.llm.invoke(prompt)
        # Handle both string and object responses
        response_text = response.content if hasattr(response, 'content') else str(response)
        variations = [query] + [v.strip() for v in response_text.split(",") if v.strip()]
        
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

| Technique | When to Use | Latency Impact | Benefit |
|-----------|-------------|----------------|---------|
| **Preprocessing** | Always | Minimal | Clean, consistent input |
| **Rewriting** | Vague/ambiguous queries | +1 LLM call | Better matches |
| **HyDE** | Complex questions | +1 LLM call + embedding | Improved retrieval |
| **Decomposition** | Multi-part questions | 2-3x searches | Comprehensive answers |
| **Hybrid Search** | Mixed queries | 2x searches | Best of both |
| **Re-ranking** | Quality-critical | +1 cross-encoder | Better precision |
| **Adaptive** | Variable queries | Variable | Optimized per query |

### Recommended Starting Point

1. **Always start** with preprocessing (it's free)
2. **Add rewriting** if users use different terminology than your docs
3. **Add hybrid search** if you have both concepts and specific terms
4. **Add re-ranking** if precision is critical
5. **Use adaptive** for complex systems with varied queries

### Common Mistakes

- **Over-optimizing**: Don't add all techniques at once
- **Ignoring latency**: Each technique adds cost; measure impact
- **No evaluation**: Test with your actual queries and documents

### Quick Decision Guide

```
What are your users asking?
      │
      ▼
┌─────────────────┐
│ Simple factual? │──No──► Try query rewriting
│ "What is X?"    │
└────────┬────────┘
         │Yes
         ▼
┌─────────────────┐
│ Simple keyword? │──Yes──► Hybrid search
│ "API key docs"  │
└────────┬────────┘
         │No
         ▼
┌─────────────────┐
│ Multi-part?     │──Yes──► Query decomposition
│ "X and Y?"      │
└────────┬────────┘
         │No
         ▼
┌─────────────────┐
│ Need precision? │──Yes──► Add re-ranking
│ Critical app    │
└────────┬────────┘
         │
         ▼
     Start simple, add as needed
```

---

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496) | 2022 | HyDE technique |
| [Query Decomposition for RAG](https://arxiv.org/abs/2510.18633) | 2025 | Multi-query approaches |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [LangChain HyDE](https://python.langchain.com/docs/retrievers/hyde) | HyDE implementation |
| [LlamaIndex Query Engine](https://docs.llamaindex.ai/en/stable/module_ides/query_engine/) | Query engine guide |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [Advanced Query Optimization](https://weaviate.io/blog/query-optimization) | Techniques |
| [HyDE Tutorial](https://zilliz.com/learn/hyde-rag) | Implementation |
| [Query Rewriting Guide](https://blog.gopenai.com/query-rewriting) | Rewriting strategies |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [texttron/hyde](https://github.com/texttron/hyde) | HyDE implementation |

---

*Previous: [Chunking Strategies](chunking-strategies.md)*

*Next: [Production Deployment Guide](./production-deployment.md)*
