# Query Rewriting for RAG

## Overview

Query rewriting transforms user queries before retrieval to improve matching with relevant documents. This technique addresses the common problem where users ask questions in ways that don't directly match how information is stored in the knowledge base.

## Why Query Rewriting Matters

```
The Query Rewriting Problem:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Query: "How do I install the package?"
                 │
                 ▼
         ┌───────────────┐
         │  Traditional  │
         │  Retrieval    │
         └───────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  ❌ No match  │  ← Document says "pip install"
         │  found        │    but user said "install"
         └───────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Query        │
         │  Rewriting    │  ← "install" → "pip install"
         └───────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  ✅ Relevant  │
         │  document     │    found!
         └───────────────┘
```

## Query Rewriting Techniques

### 1. LLM-Based Query Rewriting

Use an LLM to rewrite the query into a more retrieval-friendly form. This is the most common and versatile technique - use it for most queries where you want to improve precision.

```python
"""
LLM-Based Query Rewriting
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")

rewrite_prompt = PromptTemplate.from_template("""Rewrite this query to improve document retrieval.

Original Query: {query}

Rewrite the query to:
1. Use different synonyms where appropriate
2. Expand abbreviations
3. Make it more explicit
4. Keep it as a question

Rewritten Query:""")

class QueryRewriter:
    """Rewrite queries for better retrieval."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = rewrite_prompt
    
    def rewrite(self, query: str) -> str:
        """Rewrite a single query."""
        
        response = self.llm.invoke(
            self.prompt.format(query=query)
        )
        
        return response.content.strip()
    
    def rewrite_multi(self, query: str) -> list:
        """Generate multiple rewritten versions."""
        
        response = self.llm.invoke(f"""Generate 3 different versions of this query
that might match different relevant documents.

Original: {query}

Version 1:
Version 2:
Version 3:""")
        
        lines = response.content.strip().split('\n')
        return [l.split(':', 1)[1].strip() if ':' in l else l.strip() 
                for l in lines if l.strip()]
```

### 2. Multi-Query Retrieval

Generate multiple query variations and retrieve from all of them. This technique is useful when you want comprehensive results and are willing to handle some redundancy.

```python
"""
Multi-Query Retrieval
"""

class MultiQueryRetriever:
    """Generate and retrieve with multiple query versions."""
    
    def __init__(self, llm, base_retriever):
        self.llm = llm
        self.base_retriever = base_retriever
    
    def generate_queries(self, query: str) -> list:
        """Generate multiple query variations."""
        
        prompt = f"""Generate 5 different versions of this query that might
retrieve relevant documents. Consider different phrasings, synonyms, and perspectives.

Original: {query}

Queries:"""
        
        response = self.llm.invoke(prompt)
        
        queries = [query]  # Include original
        for line in response.content.split('\n'):
            line = line.strip()
            if line and not line.startswith(('Original', 'Queries', '-')):
                if line[0].isdigit():
                    queries.append(line.split('.', 1)[1].strip())
                elif line[0].isalpha():
                    queries.append(line.strip())
        
        return queries[:5]
    
    def retrieve(self, query: str) -> list:
        """Retrieve using multiple query versions."""
        
        queries = self.generate_queries(query)
        
        # Retrieve from each query
        all_docs = []
        for q in queries:
            docs = self.base_retriever.invoke(q)
            all_docs.extend(docs)
        
        # Deduplicate
        return self._deduplicate(all_docs)
    
    def _deduplicate(self, docs: list) -> list:
        """Remove duplicate documents."""
        seen = set()
        unique = []
        
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        
        return unique
```

### 3. Sub-Query Decomposition

Break complex queries into simpler sub-queries. Use this when you have complex, multi-part questions that need to be answered from different parts of your knowledge base.

```python
"""
Query Decomposition
"""

class QueryDecomposer:
    """Break complex queries into simpler parts."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, query: str) -> list:
        """Decompose into sub-queries."""
        
        prompt = f"""Break this complex question into simpler sub-questions
that can be answered independently.

Complex Question: {query}

Sub-questions (one per line):"""
        
        response = self.llm.invoke(prompt)
        
        sub_queries = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line[0] == '-'):
                sub_q = line.lstrip('0123456789.-').strip()
                if sub_q:
                    sub_queries.append(sub_q)
        
        return sub_queries if sub_queries else [query]
    
    def retrieve_with_decomposition(self, query: str, retriever) -> list:
        """Retrieve using decomposed sub-queries."""
        
        sub_queries = self.decompose(query)
        
        all_docs = []
        for sq in sub_queries:
            docs = retriever.invoke(sq)
            all_docs.extend(docs)
        
        return self._deduplicate(all_docs)
```

### 4. HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer document and use it for retrieval. Use HyDE when you have conceptual or meaning-based searches where exact keyword matching isn't sufficient.

```python
"""
HyDE Implementation
"""

class HyDERetriever:
    """HyDE: Hypothetical Document Embeddings."""
    
    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
    
    def generate_hypothetical(self, query: str) -> str:
        """Generate hypothetical answer document."""
        
        prompt = f"""Write a detailed hypothetical document that could answer this question.
Include relevant facts, explanations, and details that would be in a real document.

Question: {query}

Hypothetical Document:"""
        
        return self.llm.invoke(prompt).content
    
    def retrieve(self, query: str, k: int = 4):
        """Retrieve using HyDE strategy."""
        
        # Generate hypothetical document
        hypothetical = self.generate_hypothetical(query)
        
        # Get embeddings for both
        query_emb = self.embeddings.embed_query(query)
        hypo_emb = self.embeddings.embed_query(hypothetical)
        
        # Combine embeddings (average)
        combined_emb = [
            (q + h) / 2 for q, h in zip(query_emb, hypo_emb)
        ]
        
        # Search using combined embedding
        results = self.vectorstore.similarity_search_by_vector(
            combined_emb,
            k=k
        )
        
        return results, hypothetical
```

## Complete Query Rewriting Pipeline

```python
"""
Complete Query Rewriting Pipeline
"""

class AdvancedQueryRewriter:
    """Combine multiple rewriting techniques."""
    
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Initialize sub-components
        self.rewriter = QueryRewriter(llm)
        self.decomposer = QueryDecomposer(llm)
        self.hyde = HyDERetriever(vectorstore, llm)
    
    def process(self, query: str, strategy: str = "auto") -> dict:
        """
        Process query with selected strategy.
        
        Strategies:
        - "rewrite": LLM-based rewriting
        - "decompose": Query decomposition
        - "hyde": Hypothetical document embeddings
        - "multi": Multi-query retrieval
        - "auto": Let LLM decide
        """
        
        if strategy == "auto":
            strategy = self._select_strategy(query)
        
        if strategy == "rewrite":
            rewritten = self.rewriter.rewrite(query)
            docs = self.base_retriever.invoke(rewritten)
            return {"strategy": strategy, "query": rewritten, "docs": docs}
        
        elif strategy == "decompose":
            sub_queries = self.decomposer.decompose(query)
            docs = self.decomposer.retrieve_with_decomposition(
                query, self.base_retriever
            )
            return {"strategy": strategy, "sub_queries": sub_queries, "docs": docs}
        
        elif strategy == "hyde":
            docs, hypothetical = self.hyde.retrieve(query)
            return {"strategy": strategy, "hypothetical": hypothetical, "docs": docs}
        
        elif strategy == "multi":
            mq = MultiQueryRetriever(self.llm, self.base_retriever)
            queries = mq.generate_queries(query)
            docs = mq.retrieve(query)
            return {"strategy": strategy, "queries": queries, "docs": docs}
        
        else:
            # Default: no rewriting
            docs = self.base_retriever.invoke(query)
            return {"strategy": "none", "query": query, "docs": docs}
    
    def _select_strategy(self, query: str) -> str:
        """Select best strategy based on query characteristics."""
        
        prompt = f"""Analyze this query and select the best rewriting strategy.

Query: {query}

Select one:
- "rewrite": For most queries, improves precision
- "decompose": For complex, multi-part questions
- "hyde": For conceptual, meaning-based searches
- "multi": When you want comprehensive results
- "none": For already well-formed queries

Strategy:"""
        
        response = self.llm.invoke(prompt).content.strip().lower()
        
        strategies = ["rewrite", "decompose", "hyde", "multi", "none"]
        for s in strategies:
            if s in response:
                return s
        
        return "rewrite"
```

## When to Use What

| Technique | Best For | Complexity | Trade-offs |
|-----------|----------|------------|------------|
| **LLM Rewrite** | Most queries | Medium | Good precision, moderate latency |
| **Decompose** | Complex, multi-part questions | Medium | Thorough but slower |
| **HyDE** | Conceptual queries | Medium | Good semantic matching |
| **Multi-Query** | Comprehensive results | Medium | May over-retrieve |
| **None** | Well-formed queries | Low | Fastest |

## References

### Papers & Research

| Paper | Year | Link |
|-------|------|------|
| HyDE: Hypothetical Document Embeddings | 2022 | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| Query Rewriting in RAG | - | [arXiv:2405.00054](https://arxiv.org/abs/2405.00054) |

### Implementations & Tutorials

| Resource | Description |
|----------|-------------|
| [HyDE GitHub](https://github.com/texttron/hyde) | Official HyDE implementation |
| [Query Rewriting - DEV Community](https://dev.to/rogiia/build-an-advanced-rag-app-query-rewriting-h3p) | Practical tutorial |
| [Query Rewriting - Shekhar Gulati](https://shekhargulati.com/2024/07/17/query-rewriting-in-rag-applications/) | LLM-based rewriting guide |

### Blog Posts

| Blog | Description |
|------|-------------|
| [Azure AI Query Rewriting](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/raising-the-bar-for-rag-excellence-query-rewriting-and-new-semantic-ranker/4302729) | Microsoft's generative query rewriting |
| [Weaviate Advanced RAG](https://weaviate.io/blog/advanced-rag) | Query rewriting in context |
| [Part 5: Advanced RAG - Query Rewriting and HyDE](https://blog.gopenai.com/part-5-advanced-rag-techniques-llm-based-query-rewriting-and-hyde-dbcadb2f20d1) | Combined techniques |

---

*Next: [Reranking](reranking.md)*
