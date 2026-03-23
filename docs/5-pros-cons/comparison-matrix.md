# RAG Architecture Comparison

## Overview

This document provides a comprehensive comparison of different RAG architectures to help you choose the right approach for your use case. Each architecture has different tradeoffs in complexity, cost, capability, and maintenance.

## Why Architecture Choice Matters

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Architecture Selection Impact                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  YOUR DATA + QUERY TYPE                                                 │
│         │                                                               │
│         ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     DECISION POINT                                │  │
│  │                                                                   │  │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │  │
│  │   │ TEXT ONLY?  │   │RELATIONSHIPS│   │  COMPLEX    │             │  │
│  │   │    (Y/N)    │   │   NEEDED?   │   │  TASKS?     │             │  │
│  │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘             │  │
│  │          │                 │                 │                    │  │
│  │          ▼                 ▼                 ▼                    │  │
│  │   ┌──────────┐      ┌──────────┐     ┌──────────┐                 │  │
│  │   │Classic/  │      │  KG-RAG  │     │ Agentic  │                 │  │
│  │   │Multimodal│      │          │     │   RAG    │                 │  │
│  │   └──────────┘      └──────────┘     └──────────┘                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│         │                                                               │
│         ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     SYSTEM PERFORMANCE                            │  │
│  │                                                                   │  │
│  │   RIGHT ARCHITECTURE       │    WRONG ARCHITECTURE                │  │
│  │   ─────────────────────    │    ───────────────────               │  │
│  │   Better accuracy          │    Poor results                      │  │
│  │   Lower cost               │    Wasted resources                  │  │
│  │   Faster responses         │    User dissatisfaction              │  │
│  │   Easier maintenance       │    Scaling problems                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Comparison Matrix

| Feature | Classic RAG | KG-RAG | Agentic RAG | Multimodal RAG |
|---------|-------------|--------|-------------|----------------|
| **Complexity** | Low | Medium | High | High |
| **Setup Time** | Hours | Days | Days | Days |
| **Latency** | Low (ms) | Medium | Variable | Medium-High |
| **Cost** | $ | $$ | $$$ | $$$ |
| **Use Case** | Q&A, Docs | Relationships | Complex Tasks | Rich Media |
| **Reasoning** | Limited | Graph-based | Multi-step | Cross-modal |
| **Maintenance** | Easy | Moderate | Complex | Complex |
| **Scalability** | Good | Good | Medium | Medium |

## Quick Decision Guide

```
Which RAG Architecture?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start
  │
  ▼
Is your data primarily TEXT only?
  │
  ├─ Yes ──► Do you need RELATIONSHIP reasoning?
  │            │
  │            ├─ Yes ──► KG-RAG
  │            │
  │            └─ No ──► Is the query COMPLEX/MULTI-STEP?
  │                         │
  │                         ├─ Yes ──► Agentic RAG
  │                         │
  │                         └─ No ──► Classic RAG
  │
  └─ No ──► Do you have MULTIPLE modalities?
               │
               ├─ Yes ──► Multimodal RAG
               │
               └─ No ──► Classic RAG (or add multimodal later)

Quick Reference:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Query Type              | Recommended Architecture |
|-------------------------|--------------------------|
| Simple Q&A              | Classic RAG              |
| Product recommendations | KG-RAG                   |
| Research summaries      | Agentic RAG              |
| Media search            | Multimodal RAG           |
| Hybrid needs            | Agentic + KG-RAG         |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 1. Classic RAG

### Concept

**What it is**: The foundational RAG architecture that combines retrieval from a vector database with an LLM for response generation. It's a linear pipeline: query → retrieve → generate.

**Why it helps**: 
- Leverages external knowledge without fine-tuning
- Provides factual grounding for LLM responses
- Simple to implement and maintain

**When to use**:
- Document Q&A systems
- Knowledge base chatbots
- FAQ automation
- Internal search
- Simple information retrieval

**When NOT to use**:
- Complex multi-hop reasoning
- Queries requiring relationship understanding
- Rich media content
- When explainability is critical

### Implementation

```python
"""
Classic RAG Implementation
The foundational RAG pipeline: retrieve → generate
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class ClassicRAG:
    """Basic RAG pipeline for document Q&A."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        persist_directory: str = "./chroma_db"
    ):
        self.llm = ChatOllama(model=model_name, base_url=base_url)
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def query(self, question: str) -> dict:
        """Execute RAG pipeline."""
        
        # Step 1: Retrieve relevant documents
        docs = self.retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Step 2: Generate response with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": [d.page_content[:100] for d in docs],
            "num_docs": len(docs)
        }
    
    def add_documents(self, documents: list[Document]):
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)


# Usage
rag = ClassicRAG()
response = rag.query("What is RAG?")
print(response["answer"])
```

### Best For / Not For

| Best For | Not For |
|----------|---------|
| Simple Q&A | Multi-hop reasoning |
| FAQ bots | Relationship queries |
| Internal search | Rich media content |
| Knowledge bases | Complex workflows |
| Rapid prototyping | High-stakes decisions |

---

## 2. KG-RAG

### Concept

**What it is**: RAG enhanced with a knowledge graph to capture explicit relationships between entities. Uses graph traversal for multi-hop queries and provides traceable reasoning paths.

**Why it helps**:
- Handles complex relationship queries
- Provides explainable retrieval paths
- Excels at multi-hop questions (e.g., "Who is the CEO of the company that acquired X?")
- Strong for domains with structured relationships

**When to use**:
- Organizations with structured data
- Complex relationship queries
- Medical/legal/financial domains
- When explainability is required
- Product catalogs
- Organizational charts

**When NOT to use**:
- Simple Q&A without relationships
- When graph maintenance overhead is too high
- Rapid prototyping phase
- Text-heavy content without entity relationships

### Implementation

```python
"""
KG-RAG Implementation
RAG enhanced with knowledge graph for relationship reasoning
"""

from langchain_ollama import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from typing import List, Dict, Optional

class KGRetriever:
    """Retrieve from knowledge graph with text fallback."""
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm: ChatOllama
    ):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        self.llm = llm
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Hybrid retrieval: graph first, then text fallback."""
        
        # Step 1: Try graph retrieval for relationship queries
        graph_results = self._graph_retrieve(query)
        
        if graph_results:
            return self._format_graph_results(graph_results)
        
        # Step 2: Fall back to text search
        return self._text_retrieve(query, k)
    
    def _graph_retrieve(self, query: str) -> List[Dict]:
        """Query knowledge graph."""
        
        # Extract entities from query
        entities = self._extract_entities(query)
        
        if not entities:
            return []
        
        # Query relationships
        cypher = f"""
        MATCH (a)-[r]->(b)
        WHERE a.name CONTAINS '{entities[0]}' OR b.name CONTAINS '{entities[0]}'
        RETURN a.name as source, type(r) as relationship, b.name as target
        LIMIT 5
        """
        
        try:
            results = self.graph.query(cypher)
            return results
        except:
            return []
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity names from query using LLM."""
        
        prompt = f"""Extract entity names from this query that might exist in a knowledge graph.
Query: {query}

Return only the main entity names as a comma-separated list:"""
        
        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        return [e.strip() for e in text.split(",") if e.strip()]
    
    def _format_graph_results(self, results: List[Dict]) -> List[Document]:
        """Format graph results as documents."""
        
        docs = []
        for r in results:
            content = f"{r['source']} {r['relationship']} {r['target']}"
            docs.append(Document(
                page_content=content,
                metadata={"source": "knowledge_graph", "type": "relationship"}
            ))
        return docs
    
    def _text_retrieve(self, query: str, k: int) -> List[Document]:
        """Fallback to text retrieval."""
        return []


class KGRAG:
    """RAG with knowledge graph enhancement."""
    
    def __init__(self, kg_retriever, vector_retriever, llm):
        self.kg_retriever = kg_retriever
        self.vector_retriever = vector_retriever
        self.llm = llm
    
    def query(self, question: str) -> dict:
        """Query with hybrid retrieval."""
        
        # Try knowledge graph first
        kg_docs = self.kg_retriever.retrieve(question)
        
        # Also get vector results
        vector_docs = self.vector_retriever.invoke(question)
        
        # Combine and deduplicate
        all_docs = kg_docs + vector_docs
        unique_docs = self._deduplicate(all_docs)
        
        # Generate response
        context = "\n\n".join([d.page_content for d in unique_docs])
        
        prompt = f"""Based on this context, answer the question.
If the context contains relationship information, use it to provide a complete answer.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": unique_docs,
            "kg_sources": len(kg_docs),
            "vector_sources": len(vector_docs)
        }
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents."""
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:50]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique


# Usage
llm = ChatOllama(model="llama3.2")
kg_retriever = KGRetriever("bolt://localhost:7687", "neo4j", "password", llm)
# vector_retriever = ... (from Chroma)
# kg_rag = KGRAG(kg_retriever, vector_retriever, llm)
```

### Best For / Not For

| Best For | Not For |
|----------|---------|
| Relationship queries | Simple Q&A |
| Multi-hop questions | Rapid prototyping |
| Explainability needs | Text-only content |
| Structured data domains | Low-maintenance needs |
| Product/Org catalogs | When graph overhead > benefit |

---

## 3. Agentic RAG

### Concept

**What it is**: RAG enhanced with autonomous agents that can make decisions about retrieval strategy, use multiple tools, iterate on results, and self-correct when initial attempts fail.

**Why it helps**:
- Adapts retrieval strategy per query
- Handles complex, multi-step tasks
- Self-correction through reflection
- Can use multiple data sources and tools
- Higher quality for complex queries

**When to use**:
- Complex research tasks
- Multi-step analysis
- Dynamic information needs
- When quality is critical
- Adaptive workflows
- Multi-source queries

**When NOT to use**:
- Simple, low-latency needs
- Cost-sensitive applications
- Single-lookup queries
- When transparency is required (agent decisions are hard to trace)

### Implementation

```python
"""
Agentic RAG Implementation
RAG with autonomous agents for dynamic retrieval and self-correction
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json

class AgentAction(Enum):
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    REFINE = "refine"
    REJECT = "reject"
    USE_TOOL = "use_tool"

@dataclass
class AgentDecision:
    action: AgentAction
    reasoning: str
    params: Dict

class AgenticRetriever:
    """Agent that decides retrieval strategy dynamically."""
    
    def __init__(self, llm, vector_retriever, tools: List[Callable] = None):
        self.llm = llm
        self.vector_retriever = vector_retriever
        self.tools = tools or []
        self.max_iterations = 3
    
    def decide(self, query: str, context: str = "") -> AgentDecision:
        """LLM decides next action."""
        
        tools_desc = "\n".join([f"- {t.__name__}: {t.__doc__}" for t in self.tools]) if self.tools else "No tools available"
        
        prompt = f"""Analyze this query and decide the next action.

Query: {query}

Current Context: {context if context else "No context yet"}

Available Tools:
{tools_desc}

Actions available:
- retrieve: Get more documents from vector store
- generate: Generate final answer with current context
- refine: Improve query and try again
- use_tool: Use a specific tool
- reject: Cannot answer this query

Respond in JSON format:
{{
    "action": "action_name",
    "reasoning": "why you chose this action",
    "params": {{"any additional parameters}}
}}"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        
        try:
            decision = json.loads(text)
            return AgentDecision(
                action=AgentAction(decision["action"]),
                reasoning=decision.get("reasoning", ""),
                params=decision.get("params", {})
            )
        except:
            return AgentDecision(AgentAction.RETRIEVE, "default", {})
    
    def execute(self, query: str) -> Dict:
        """Execute agent loop."""
        
        all_docs = []
        current_context = ""
        
        for iteration in range(self.max_iterations):
            decision = self.decide(query, current_context)
            
            if decision.action == AgentAction.RETRIEVE:
                docs = self.vector_retriever.invoke(query)
                all_docs.extend(docs)
                current_context = "\n\n".join([d.page_content for d in all_docs])
                
            elif decision.action == AgentAction.GENERATE:
                return self._generate(query, all_docs)
            
            elif decision.action == AgentAction.REFINE:
                refined = self._refine_query(query)
                query = refined
                
            elif decision.action == AgentAction.USE_TOOL:
                if self.tools and decision.params.get("tool_name"):
                    tool = next((t for t in self.tools if t.__name__ == decision.params["tool_name"]), None)
                    if tool:
                        result = tool(decision.params.get("args", []))
                        all_docs.extend(result)
            
            elif decision.action == AgentAction.REJECT:
                return {
                    "answer": "I cannot answer this query with the available information.",
                    "sources": [],
                    "iterations": iteration + 1,
                    "reasoning": decision.reasoning
                }
        
        # Max iterations reached, generate what we have
        return self._generate(query, all_docs)
    
    def _generate(self, query: str, docs: List) -> Dict:
        """Generate final response."""
        
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""Based on this context, answer the question thoroughly.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": [d.page_content[:100] for d in docs],
            "num_sources": len(docs)
        }
    
    def _refine_query(self, query: str) -> str:
        """Refine query for better retrieval."""
        
        prompt = f"""Rewrite this query to be more effective for document retrieval.
Make it more specific and include relevant keywords.

Original: {query}

Refined:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


class SelfCorrectingRAG:
    """RAG with self-correction capability."""
    
    def __init__(self, agentic_retriever, llm):
        self.retriever = agentic_retriever
        self.llm = llm
    
    def query(self, question: str) -> Dict:
        """Query with automatic verification."""
        
        # Get initial response
        result = self.retriever.execute(question)
        
        # Verify response quality
        verified = self._verify(question, result["answer"], result.get("sources", []))
        
        if not verified["quality_ok"]:
            # Try to improve
            improved = self._improve(question, result, verified["issues"])
            return improved
        
        return result
    
    def _verify(self, question: str, answer: str, sources: List) -> Dict:
        """Verify answer quality."""
        
        prompt = f"""Evaluate this answer for quality.

Question: {question}
Answer: {answer}
Sources: {len(sources)} documents

Respond with:
{{
    "quality_ok": true/false,
    "issues": ["list of issues if any"],
    "hallucination_risk": "low/medium/high"
}}"""
        
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            return {"quality_ok": True, "issues": [], "hallucination_risk": "low"}
    
    def _improve(self, question: str, result: Dict, issues: List) -> Dict:
        """Attempt to improve response."""
        
        # Could retry with different parameters, expand context, etc.
        result["verified"] = False
        result["issues"] = issues
        return result


# Usage
llm = ChatOllama(model="llama3.2")
# vector_retriever = ... 
# agent = AgenticRetriever(llm, vector_retriever)
# response = agent.execute("What is the relationship between X and Y?")
```

### Best For / Not For

| Best For | Not For |
|----------|---------|
| Complex research | Simple Q&A |
| Multi-step analysis | Cost-sensitive apps |
| Dynamic information needs | Low-latency requirements |
| High-quality requirements | Single-source queries |
| Adaptive workflows | Transparent/auditable needs |

---

## 4. Multimodal RAG

### Concept

**What it is**: RAG that processes and retrieves across multiple modalities (text, images, audio, video) using specialized encoders and unified embedding spaces.

**Why it helps**:
- Handles rich media content
- Cross-modal retrieval (query text, find images)
- Unifies content types in single pipeline
- Essential for multimedia applications

**When to use**:
- Rich media archives
- Video/podcast search
- Enterprise content (slides, screenshots)
- Educational platforms
- Media analysis
- Customer support with screenshots

**When NOT to use**:
- Text-only content
- Low-latency requirements
- Limited compute resources
- Simple document Q&A

### Implementation

```python
"""
Multimodal RAG Implementation
RAG that handles text, images, audio, and video
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import Union, List, Dict, Optional
import base64
import os

class MultimodalEncoder:
    """Encode different modalities into embeddings."""
    
    def __init__(self, text_model: str = "nomic-embed-text"):
        self.text_embeddings = OllamaEmbeddings(model=text_model)
        self.image_model = "llama3.2"  # or specialized vision model
    
    def encode_text(self, text: str) -> List[float]:
        """Encode text to embedding."""
        return self.text_embeddings.embed_query(text)
    
    def encode_image(self, image_path: str) -> List[float]:
        """Encode image to embedding.
        
        Note: In production, use dedicated vision models like:
        - CLIP (OpenAI)
        - SigLIP (Google)
        - BLIP-2
        """
        # Placeholder - in production use vision model API
        # This would call a vision model's embedding endpoint
        text_desc = self._describe_image(image_path)
        return self.text_embeddings.embed_query(text_desc)
    
    def encode_audio(self, audio_path: str) -> List[float]:
        """Encode audio/transcription to embedding."""
        # For audio, first transcribe then embed
        # Use Whisper or similar for transcription
        text = self._transcribe_audio(audio_path)
        return self.text_embeddings.embed_query(text)
    
    def encode_video(self, video_path: str) -> List[float]:
        """Encode video to embedding."""
        # Extract frames or use video transcription
        frames = self._extract_key_frames(video_path)
        descriptions = [self._describe_image(f) for f in frames]
        combined = " | ".join(descriptions)
        return self.text_embeddings.embed_query(combined)
    
    def encode(self, content: Union[str, bytes]) -> List[float]:
        """Auto-detect and encode content type."""
        
        if isinstance(content, str):
            if os.path.isfile(content):
                ext = content.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg', 'png', 'gif']:
                    return self.encode_image(content)
                elif ext in ['mp3', 'wav', 'm4a']:
                    return self.encode_audio(content)
                elif ext in ['mp4', 'avi', 'mov']:
                    return self.encode_video(content)
            
            # Plain text
            return self.encode_text(content)
        
        # For bytes, assume image
        return self.encode_text("[image content]")
    
    def _describe_image(self, path: str) -> str:
        """Describe image using vision model."""
        # Placeholder - use actual vision model
        return f"Image from {os.path.basename(path)}"
    
    def _transcribe_audio(self, path: str) -> str:
        """Transcribe audio file."""
        # Placeholder - use Whisper
        return f"Audio transcription from {os.path.basename(path)}"
    
    def _extract_key_frames(self, path: str) -> List[str]:
        """Extract key frames from video."""
        # Placeholder
        return []


class MultimodalVectorStore:
    """Vector store that handles multiple content types."""
    
    def __init__(self, encoder: MultimodalEncoder, persist_dir: str = "./multimodal_db"):
        self.encoder = encoder
        self.text_store = Chroma(persist_directory=persist_dir + "_text")
    
    def add_documents(self, documents: List[Document]):
        """Add documents with automatic modality detection."""
        
        for doc in documents:
            content = doc.page_content
            embedding = self.encoder.encode(content)
            self.text_store.add_texts(
                texts=[content],
                metadatas=[doc.metadata],
                embeddings=[embedding]
            )
    
    def add_file(self, file_path: str, metadata: Dict = None):
        """Add a file (image, audio, video, text)."""
        
        embedding = self.encoder.encode(file_path)
        
        # Store reference
        self.text_store.add_texts(
            texts=[file_path],
            metadatas=[metadata or {}],
            embeddings=[embedding]
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search across all modalities."""
        query_emb = self.encoder.encode_text(query)
        return self.text_store.similarity_search_by_vector(query_emb, k=k)


class MultimodalRAG:
    """Complete multimodal RAG system."""
    
    def __init__(
        self,
        text_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text"
    ):
        self.llm = ChatOllama(model=text_model)
        self.encoder = MultimodalEncoder(embedding_model)
        self.vectorstore = MultimodalVectorStore(self.encoder)
        self.retriever = self.vectorstore.text_store.as_retriever(search_kwargs={"k": 3})
    
    def query(self, query: str) -> Dict:
        """Query with multimodal understanding."""
        
        # If query contains image reference, handle accordingly
        # For now, assume text query
        
        docs = self.retriever.invoke(query)
        
        # Prepare context with modality indicators
        context_parts = []
        for doc in docs:
            modality = doc.metadata.get("modality", "text")
            if modality != "text":
                context_parts.append(f"[{modality.upper()}] {doc.page_content}")
            else:
                context_parts.append(doc.page_content)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on this context (which may include text, images, audio, or video), answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": [d.page_content[:100] for d in docs],
            "modalities": [d.metadata.get("modality", "text") for d in docs]
        }
    
    def add_content(self, file_path: str, metadata: Dict = None):
        """Add content of any supported type."""
        metadata = metadata or {}
        
        # Detect modality
        ext = file_path.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            metadata["modality"] = "image"
        elif ext in ['mp3', 'wav', 'm4a', 'flac']:
            metadata["modality"] = "audio"
        elif ext in ['mp4', 'avi', 'mov', 'mkv']:
            metadata["modality"] = "video"
        else:
            metadata["modality"] = "text"
        
        self.vectorstore.add_file(file_path, metadata)


# Usage
# mm_rag = MultimodalRAG()
# mm_rag.add_content("image.png", {"source": "user_upload"})
# response = mm_rag.query("What's in the image?")
```

### Best For / Not For

| Best For | Not For |
|----------|---------|
| Rich media archives | Text-only content |
| Video/podcast search | Low-latency requirements |
| Enterprise content | Limited compute |
| Educational platforms | Simple document Q&A |
| Media analysis | When single modality sufficient |

---

## Detailed Comparisons

### Classic RAG vs KG-RAG

| Aspect | Classic RAG | KG-RAG |
|--------|-------------|--------|
| **Data Structure** | Unstructured text | Graph + text |
| **Relationship Handling** | Implicit in embeddings | Explicit graph edges |
| **Multi-hop Queries** | ❌ Limited | ✅ Strong |
| **Explainability** | Black box | Traceable paths |
| **Setup Complexity** | Simple | Requires KG DB |
| **Maintenance** | Easy | Moderate |

### Classic RAG vs Agentic RAG

| Aspect | Classic RAG | Agentic RAG |
|--------|-------------|------------|
| **Workflow** | Linear | Dynamic |
| **Retrieval** | Single pass | Iterative |
| **Adaptability** | Fixed | Learns/Adjusts |
| **Error Handling** | None | Self-correction |
| **Tool Use** | Single | Multiple |

### Classic RAG vs Multimodal RAG

| Aspect | Classic RAG | Multimodal RAG |
|--------|-------------|----------------|
| **Input Types** | Text only | Text + images + audio + video |
| **Processing** | Single pipeline | Multiple pipelines |
| **Embeddings** | Text only | Multimodal models |
| **Storage** | Vector DB | Multi-type stores |

---

## Performance Comparison

### Query Complexity Handling

| Query Type | Classic | KG | Agentic | Multimodal |
|------------|---------|-----|---------|------------|
| Simple fact | ✅ | ✅ | ✅ | ✅ |
| List items | ✅ | ✅ | ✅ | ✅ |
| Single hop | ✅ | ✅ | ✅ | ✅ |
| Multi-hop | ❌ | ✅ | ✅ | ✅ |
| Comparison | ❌ | ✅ | ✅ | ✅ |
| Temporal | ❌ | ✅ | ✅ | ✅ |
| Cross-modal | ❌ | ❌ | ❌ | ✅ |

### Accuracy by Use Case

| Use Case | Classic | KG | Agentic | Multimodal |
|----------|---------|-----|---------|------------|
| FAQ | 85-90% | 85-90% | 90-95% | 85-90% |
| Research | 70-80% | 75-85% | 85-95% | 70-80% |
| Domain-specific | 75-85% | 85-95% | 85-95% | 75-85% |
| Multi-document | 65-75% | 70-80% | 80-90% | 65-75% |

---

## Cost Comparison (Monthly Estimates)

| Architecture | Setup | Monthly (1000 queries/day) |
|--------------|-------|----------------------------|
| Classic RAG | $500 | $200-500 |
| KG-RAG | $1,500 | $500-1,000 |
| Agentic RAG | $2,000 | $1,000-3,000 |
| Multimodal RAG | $3,000 | $2,000-5,000 |

*Estimates based on cloud infrastructure + API costs*

---

## Migration Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RAG Architecture Evolution Path                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   START HERE                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐                                                           │
│  │ Classic RAG  │                                                           │
│  │  (Hours)     │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         │ Need relationships?                                               │
│         │                                                                   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   KG-RAG     │ ─────────────────────────────────────┐                    │
│  │  (Days)      │  Add complex │                       │                    │
│  └──────┬───────┘  workflows?  │                       │                    │
│         │                      │                       │                    │
│         │                      │                       │                    │
│         │                      No                     Yes                   │
│         │                      │                       │                    │
│         ▼                      ▼                       ▼                    │
│  ┌──────────────┐       ┌──────────────┐        ┌──────────────┐            │
│  │  Keep KG-RAG │       │ Agentic RAG  │        │ Agentic RAG  │            │
│  │    only      │       │   (Days)     │        │   + KG-RAG   │            │
│  └──────────────┘       └──────┬───────┘        └──────┬───────┘            │
│                                │                       │                    │
│                                │ Need rich media?      │                    │
│                                │                       │                    │
│                                │                       │                    │
│                                ▼                       │                    │
│                         ┌──────────────┐               │                    │
│                         │ Multimodal   │  ◄────────────┘                    │
│                         │    RAG       │                                    │
│                         └──────────────┘                                    │
│                                                                             │
│   EVOLUTION OPTIONS:                                                        │
│   ─────────────────                                                         │
│   Classic → KG-RAG → Agentic → Multimodal  (Full evolution)                 │
│   Classic → KG-RAG                         (Add relationships)              │
│   Classic → Agentic                        (Add complex workflows)          │
│   Classic → Multimodal                     (Add media support)              │
│   Classic                                  (Stay simple)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Architecture | Complexity | Cost | Best Use Case | Setup Time |
|--------------|------------|------|----------------|------------|
| **Classic RAG** | Low | $ | Simple Q&A, Docs | Hours |
| **KG-RAG** | Medium | $$ | Relationships, Multi-hop | Days |
| **Agentic RAG** | High | $$$ | Complex tasks, Research | Days |
| **Multimodal RAG** | High | $$$ | Rich media, Cross-modal | Days |

---

## Common Mistakes

| Mistake | Why It's Bad | Fix |
|---------|--------------|-----|
| **Over-engineering** | Using Agentic when Classic suffices | Start simple, add complexity as needed |
| **Ignoring relationships** | Using Classic for relationship queries | Use KG-RAG |
| **No modality planning** | Adding multimodal later is hard | Plan upfront if media content expected |
| **Skipping evaluation** | Can't measure improvement | Use RAGAS, ARES benchmarks |
| **Ignoring cost** | Agentic can be 10x more expensive | Set budgets, monitor costs |

---

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [A Systematic Literature Review of RAG](https://arxiv.org/abs/2508.06401) | 2025 | Comprehensive overview |
| [Agentic RAG: A Survey](https://arxiv.org/abs/2501.09136) | 2025 | Agentic systems |
| [Comprehensive RAG Survey](https://arxiv.org/abs/2506.00054) | 2025 | Architectures & enhancements |
| [Multimodal RAG Survey](https://arxiv.org/abs/2502.08826) | 2025 | Beyond text |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Production frameworks |
| [LlamaIndex](https://www.llamaindex.ai/) | Data indexing |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | Knowledge graph RAG |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG evaluation |
| [ARES](https://github.com/stanford-futuredata/ARES) | Automated evaluation |

---

*Previous: [Research Directions](research-directions.md)*

*Next: [Evaluation Metrics](../3-technical/evaluation-metrics.md)*
