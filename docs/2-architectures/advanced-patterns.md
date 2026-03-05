# Advanced RAG Patterns

## Overview

This document covers advanced RAG patterns that go beyond the basic retrieve-then-generate architecture. These patterns address specific challenges like retrieval quality, error handling, and complex reasoning.

## Pattern Taxonomy

```
Advanced RAG Patterns:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                        Advanced Patterns                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Query          │  │  Retrieval      │  │  Generation     │          │
│  │  Enhancement    │  │  Enhancement    │  │  Enhancement    │          │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤          │
│  │ • HyDE          │  │ • Corrective    │  │ • Self-RAG      │          │
│  │ • Query         │  │   RAG           │  │ • Chain-of-Note │          │
│  │   Rewriting     │  │ • Active RAG    │  │ • Corrective    │          │
│  │ • Query         │  │ • REPLUG        │  │   Generation    │          │
│  │   Decomposition │  │                 │  │                 │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Pipeline       │  │  Knowledge      │  │  Evaluation     │          │
│  │  Patterns       │  │  Enhancement    │  │  & Debugging    │          │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤          │
│  │ • Router        │  │ • GraphRAG      │  │ • Adaptive      │          │
│  │ • Fallback      │  │ • RouteRAG      │  │   RAG           │          │
│  │ • Parallel      │  │ • Merge-KG      │  │ • Human-in-loop │          │
│  │ • Iterative     │  │                 │  │                 │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. HyDE (Hypothetical Document Embeddings)

### Concept

Generate a hypothetical answer document, then use it for retrieval instead of the original query.

```python
"""
HyDE Implementation
"""

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma

class HyDERetriever:
    """HyDE: Hypothetical Document Embeddings."""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def generate_hypothetical(self, query: str) -> str:
        """Generate hypothetical answer document."""
        
        prompt = f"""Write a hypothetical document that could answer this question.
Include detailed information that would be relevant.

Question: {query}

Hypothetical Document:"""

        return self.llm.invoke(prompt)
    
    def retrieve(self, query: str, k: int = 4):
        """Retrieve using HyDE strategy."""
        
        # Generate hypothetical document
        hypothetical = self.generate_hypothetical(query)
        
        # Get embeddings
        query_emb = self.vectorstore.embedding.embed_query(query)
        hypo_emb = self.vectorstore.embedding.embed_query(hypothetical)
        
        # Combine embeddings (average)
        combined_emb = [(q + h) / 2 for q, h in zip(query_emb, hypo_emb)]
        
        # Search
        results = self.vectorstore.vectorstore.similarity_search_by_vector(
            combined_emb,
            k=k
        )
        
        return results
```

### When to Use HyDE

- Complex, conceptual queries
- When users ask in different ways than documents are written
- For questions requiring explanatory answers

## 2. Self-RAG

### Concept

Train or prompt the model to decide when to retrieve and how to critique its own output.

```python
"""
Self-RAG Implementation (Reflection-based)
"""

class SelfRAGChain:
    """Self-Reflective RAG with evaluation."""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def should_retrieve(self, query: str) -> bool:
        """Decide if retrieval is needed."""
        
        prompt = f"""Question: {query}

Does this question require external knowledge to answer accurately?
Consider:
- Is it asking for recent information?
- Is it asking for specific facts?
- Could the answer be in your training data?

Answer yes or no:"""

        response = self.llm.invoke(prompt).content.strip().lower()
        return "yes" in response
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Evaluate if answer is grounded in context."""
        
        prompt = f"""Context: {context}

Answer: {answer}

On a scale of 0-10, how much is the answer supported by the context?
Only respond with a number:"""

        return float(self.llm.invoke(prompt).content.strip()) / 10
    
    def run(self, query: str) -> str:
        """Execute Self-RAG."""
        
        # Decision: retrieve or not
        if not self.should_retrieve(query):
            return self.llm.invoke(f"Answer this question: {query}").content
        
        # Retrieve
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate
        answer = self.llm.invoke(
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        
        # Evaluate
        score = self.evaluate_faithfulness(answer, context)
        
        # Refine if needed
        if score < 0.7:
            # Try with more context or reformulate
            refined = self.llm.invoke(
                f"The previous answer had low context support.\n"
                f"Context: {context}\n\nQuestion: {query}\n\nProvide a better answer:"
            )
            return refined
        
        return answer
```

## 3. Corrective RAG

### Concept

Detect and correct retrieval or generation errors during the pipeline.

```python
"""
Corrective RAG Implementation
"""

class CorrectiveRAG:
    """Detect and correct errors in RAG pipeline."""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def check_retrieval(self, query: str, docs: list) -> dict:
        """Check retrieval quality."""
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Evaluate this retrieval:

Query: {query}

Retrieved Context: {context}

Rate from 1-5:
1. Relevance: How relevant are the retrieved documents?
2. Sufficiency: Does the context answer the question?
3. Quality: Is the context high quality?

Respond as JSON:"""

        response = self.llm.invoke(prompt)
        return json.loads(response)
    
    def correct_retrieval(self, query: str, docs: list) -> list:
        """Correct poor retrieval results."""
        
        assessment = self.check_retrieval(query, docs)
        
        if assessment.get("relevance", 5) < 3:
            # Try alternative retrieval
            # - Use different search type
            # - Expand query
            # - Use hybrid search
            return self._alternative_retrieval(query)
        
        return docs
    
    def _alternative_retrieval(self, query: str) -> list:
        """Try alternative retrieval strategies."""
        
        # Try expanding query
        expanded = self.llm.invoke(
            f"Expand this query with synonyms: {query}"
        )
        
        docs = self.retriever.get_relevant_documents(expanded)
        
        if not docs:
            # Try general search
            docs = self.retriever.get_relevant_documents(query)
        
        return docs
    
    def check_generation(self, query: str, answer: str, context: str) -> dict:
        """Check generation quality."""
        
        prompt = f"""Evaluate this answer:

Question: {query}
Context: {context}
Answer: {answer}

Check for:
1. Hallucinations: Does answer contain unsupported info?
2. Completeness: Does it fully answer the question?
3. Coherence: Is it well-structured?

Respond as JSON:"""

        return json.loads(self.llm.invoke(prompt).content)
    
    def run(self, query: str) -> str:
        """Execute Corrective RAG with checks."""
        
        # Retrieve
        docs = self.retriever.get_relevant_documents(query)
        
        # Check and correct retrieval
        docs = self.correct_retrieval(query, docs)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate
        answer = self.llm.invoke(
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        
        # Check generation
        gen_check = self.check_generation(query, answer, context)
        
        # Correct if needed
        if gen_check.get("hallucinations", 0) > 2:
            answer = self._regenerate(query, context, answer)
        
        return answer
    
    def _regenerate(self, query: str, context: str, bad_answer: str) -> str:
        """Regenerate with correction prompt."""
        
        prompt = f"""The previous answer had issues.

Question: {query}
Context: {context}
Bad Answer: {bad_answer}

Provide a corrected answer that:
- Only uses information from context
- Directly answers the question
- Is well-structured:"""

        return self.llm.invoke(prompt)
```

## 4. GraphRAG (Microsoft)

### Concept

Build a knowledge graph from documents and use community summaries for retrieval.

```python
"""
GraphRAG Implementation
"""

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

class GraphRAG:
    """GraphRAG with community summarization."""
    
    def __init__(self, documents, llm, graph_config):
        self.llm = llm
        self.graph = Neo4jGraph(**graph_config)
        self.graph_transformer = LLMGraphTransformer(llm=llm)
        
        # Build graph
        self._build_graph(documents)
    
    def _build_graph(self, documents):
        """Build knowledge graph from documents."""
        
        # Extract graph
        graph_docs = self.graph_transformer.convert_to_graph_documents(documents)
        
        # Store in Neo4j
        self.graph.add_graph_documents(graph_docs)
        
        # Create community summaries
        self._create_communities()
    
    def _create_communities(self):
        """Create community summaries using graph algorithms."""
        
        # Run community detection
        query = """
        CALL algo.louvain(null, null, {
            write:true,
            writeProperty:'community'
        })
        """
        self.graph.query(query)
        
        # Generate summaries per community
        communities = self.graph.query("""
            MATCH (n)
            RETURN n.community as community, collect(n) as nodes
        """)
        
        for community in communities:
            # Summarize each community
            summary = self.llm.invoke(
                f"Summarize this knowledge community: {community['nodes']}"
            )
            # Store summary
            self.graph.query("""
                MATCH (n)
                WHERE n.community = $community
                SET n.community_summary = $summary
            """, {"community": community["community"], "summary": summary})
    
    def retrieve_global(self, query: str) -> str:
        """Global search using community summaries."""
        
        # Get relevant communities
        communities = self.graph.query("""
            MATCH (n)
            WHERE n.community_summary CONTAINS $query
            RETURN n.community_summary as summary
            LIMIT 5
        """, {"query": query})
        
        context = "\n\n".join([c["summary"] for c in communities])
        
        # Generate answer
        return self.llm.invoke(f"""
            Context from multiple knowledge sources:
            {context}
            
            Question: {query}
            
            Comprehensive answer:"""
        })
    
    def retrieve_local(self, query: str) -> str:
        """Local search using specific entities."""
        
        # Find relevant entities
        entities = self.graph.query("""
            MATCH (n)
            WHERE n.name CONTAINS $query OR n.description CONTAINS $query
            RETURN n
            LIMIT 10
        """, {"query": query})
        
        # Get neighborhood
        context_parts = []
        for entity in entities:
            neighbors = self.graph.query(f"""
                MATCH (e {{{entity['n']['name']}}})-[r]-(related)
                RETURN e, r, related
            """)
            context_parts.extend([str(n) for n in neighbors])
        
        context = "\n\n".join(context_parts)
        
        return self.llm.invoke(f"""
            Context: {context}
            
            Question: {query}
            
            Answer:"""
        )
    
    def run(self, query: str, mode: str = "local") -> str:
        """Execute GraphRAG."""
        
        if mode == "global":
            return self.retrieve_global(query)
        else:
            return self.retrieve_local(query)
```

## 5. Router Pattern

### Concept

Dynamically route queries to different retrieval or processing strategies.

```python
"""
Query Router Implementation
"""

from enum import Enum

class RetrievalMode(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    KG = "knowledge_graph"
    WEB = "web_search"
    HYBRID = "hybrid"

class QueryRouter:
    """Route queries to appropriate retrieval strategy."""
    
    def __init__(self, retrievers: dict, llm):
        self.retrievers = retrievers
        self.llm = llm
    
    def determine_mode(self, query: str) -> RetrievalMode:
        """Determine best retrieval mode."""
        
        prompt = f"""Analyze this query and select the best retrieval strategy.

Query: {query}

Options:
- semantic: General questions, conceptual searches
- keyword: Specific terms, names, codes
- knowledge_graph: Questions about relationships
- web_search: Current events, real-time info
- hybrid: Complex questions needing multiple approaches

Respond with just one word:"""

        response = self.llm.invoke(prompt).content.strip().lower()
        
        mode_map = {
            "semantic": RetrievalMode.SEMANTIC,
            "keyword": RetrievalMode.KEYWORD,
            "knowledge_graph": RetrievalMode.KG,
            "knowledgegraph": RetrievalMode.KG,
            "web_search": RetrievalMode.WEB,
            "websearch": RetrievalMode.WEB,
            "hybrid": RetrievalMode.HYBRID,
        }
        
        return mode_map.get(response, RetrievalMode.SEMANTIC)
    
    def retrieve(self, query: str):
        """Execute routed retrieval."""
        
        mode = self.determine_mode(query)
        
        retriever = self.retrievers.get(mode.value, self.retrievers["semantic"])
        
        return retriever.get_relevant_documents(query)
    
    def retrieve_parallel(self, query: str) -> list:
        """Parallel retrieval from multiple sources."""
        
        results = []
        
        for mode, retriever in self.retrievers.items():
            docs = retriever.get_relevant_documents(query)
            results.extend(docs)
        
        # Deduplicate and rerank
        return self._deduplicate(results)
    
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

## 6. Iterative Refinement

### Concept

Repeatedly retrieve and refine until satisfactory answer is generated.

```python
"""
Iterative RAG with Refinement
"""

class IterativeRAG:
    """Iteratively refine retrieval and generation."""
    
    def __init__(self, llm, retriever, max_iterations=3):
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations
    
    def run(self, query: str) -> str:
        """Execute iterative refinement."""
        
        all_context = []
        
        for iteration in range(self.max_iterations):
            # Retrieve
            docs = self.retriever.get_relevant_documents(query)
            new_context = [doc.page_content for doc in docs]
            
            # Check if we have enough information
            if iteration > 0:
                enough = self._check_sufficiency(
                    query, 
                    all_context + new_context
                )
                if enough:
                    break
            
            all_context.extend(new_context)
            
            # Generate partial answer for feedback
            partial = self.llm.invoke(
                f"Context:\n{chr(10).join(all_context)}\n\n"
                f"Question: {query}\n\n"
                f"Partial answer so far:"
            )
            
            # Get feedback for next iteration
            missing = self._get_missing_info(query, partial)
            
            if not missing:
                break
            
            # Update query for next iteration
            query = f"{query} Also need: {missing}"
        
        # Final answer
        return self.llm.invoke(
            f"Context:\n{chr(10).join(all_context)}\n\n"
            f"Question: {query}\n\n"
            f"Final answer:"
        )
    
    def _check_sufficiency(self, query: str, context: list) -> bool:
        """Check if we have enough context."""
        
        prompt = f"""Question: {query}

Context:
{chr(10).join(context[:3])}

Can this question be fully answered with the above context?
Yes or no:"""

        response = self.llm.invoke(prompt).content.strip().lower()
        return "yes" in response
    
    def _get_missing_info(self, query: str, partial_answer: str) -> str:
        """Get feedback on what's missing."""
        
        prompt = f"""Question: {query}

Partial Answer: {partial_answer}

What information is still missing or incomplete?
Briefly describe:"""

        response = self.llm.invoke(prompt).strip()
        
        if "nothing" in response.lower() or "complete" in response.lower():
            return ""
        
        return response
```

## Summary Table

| Pattern | Purpose | Complexity | Best For |
|---------|---------|------------|----------|
| **HyDE** | Better retrieval via hypothetical docs | Medium | Conceptual queries |
| **Self-RAG** | Model reflects on retrieval need | High | Quality-critical |
| **Corrective RAG** | Detect and fix errors | High | Production systems |
| **GraphRAG** | Graph-based knowledge retrieval | High | Large documents |
| **Router** | Dynamic strategy selection | Medium | Variable queries |
| **Iterative** | Refine until quality | High | Complex questions |

---

*Next: [Vector Databases Guide](./vector-databases.md)*
