# What is Retrieval-Augmented Generation (RAG)?

## Overview

**Retrieval-Augmented Generation (RAG)** is an AI framework that enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge bases before generating responses. This combination of retrieval and generation addresses key limitations of traditional LLMs.

## Core Concept

RAG works by:

1. **Retrieval**: Given a user query, search a knowledge base for relevant documents or information
2. **Augmentation**: Combine the retrieved context with the original query
3. **Generation**: Use an LLM to generate a response grounded in the retrieved information

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User Query                                                │
│       │                                                     │
│       ▼                                                     │
│   ┌─────────────┐                                           │
│   │  Retrieval  │ ◄──── External Knowledge Base             │
│   │   Engine    │      (Vector DB +/ Knowledge Graph)       │
│   └──────┬──────┘                                           │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────┐     ┌─────────────┐                       │
│   │  Retrieved  │────►│     LLM     │                       │
│   │   Context   │     │  Generation │                       │
│   └─────────────┘     └──────┬──────┘                       │
│                              │                              │
│                              ▼                              │
│                       ┌─────────────┐                       │
│                       │  Generated  │                       │
│                       │   Response  │                       │
│                       └─────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Why RAG Matters

### The LLM Limitation Problem

Traditional LLMs have three fundamental limitations:

| Limitation                | Description                                     | Impact                                      |
|---------------------------|-------------------------------------------------|---------------------------------------------|
| **Knowledge Cutoff**      | Training data has a fixed date                  | Cannot answer questions about recent events |
| **Hallucinations**        | Model generates plausible but false information | Unreliable for factual queries              |
| **No Source Attribution** | Cannot cite or reference sources                | Hard to verify responses                    |

### How RAG Addresses These

1. **Fresh Knowledge**: Retrieved in real-time from up-to-date sources
2. **Grounded Responses**: Generated content is tied to actual documents
3. **Traceability**: Can cite sources for verification

## Types of RAG

This learning resource covers multiple RAG architectures:

1. **Classic RAG** (Naive RAG)
   - Basic retrieve-then-generate pipeline
   - Best for: Simple Q&A, document chatbots

2. **Knowledge Graph RAG (KG-RAG)**
   - Uses structured knowledge graphs for retrieval
   - Best for: Complex relationships, domain-specific reasoning

3. **Agentic RAG**
   - Autonomous agents that plan and execute retrieval
   - Best for: Multi-step tasks, dynamic workflows

4. **Multimodal RAG**
   - Handles text, images, video, audio
   - Best for: Rich media analysis, cross-modal queries

5. **Advanced Patterns**
   - Self-RAG, HyDE, Corrective RAG, GraphRAG
   - Research-focused implementations

## Key Components

### 1. Document Processing
- Loading documents from various sources
- Text extraction (PDF, HTML, Markdown)
- Metadata extraction

### 2. Text Chunking
- Splitting documents into manageable pieces
- Various strategies: fixed size, semantic, recursive

### 3. Embedding Generation
- Converting text chunks into vector representations
- Models: OpenAI, Cohere, sentence-transformers, BGE

### 4. Vector Database
- Storing and indexing embeddings
- Options: Pinecone, Weaviate, Chroma, Milvus, pgvector

### 5. Retrieval Strategies
- Semantic search (dense retrieval)
- Keyword search (BM25)
- Hybrid search

### 6. Generation
- LLM integration
- Prompt engineering
- Context window management

## When to Use RAG

RAG is ideal when you need:

✅ Up-to-date information retrieval  
✅ Domain-specific knowledge  
✅ Source citation and verification  
✅ Reducing hallucinations  
✅ Private/internal data access  

RAG may not be needed when:

- General conversation without specific knowledge requirements
- Creative writing tasks
- Tasks where training data is sufficient

## Quick Example

```python
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Create vector store from documents (using Ollama - free, local)
vectorstore = Chroma.from_documents(documents, OllamaEmbeddings(model="nomic-embed-text"))

# Create QA chain (using Ollama - free, local)
qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"),
    retriever=vectorstore.as_retriever()
)

# Query
result = qa.run("What are the key benefits of RAG?")
```

## Next Steps

- [Why RAG?](why-rag.md) - Deep dive into the motivation
- [Classic RAG Implementation](../2-architectures/classic-rag/) - Build your first RAG
- [Technical Deep Dive](../3-technical/) - Understand the components

---

*See also: [Evolution of RAG](evolution-of-rag.md)*
