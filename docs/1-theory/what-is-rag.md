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
│   │   Engine    │      (Vector DB and/or Knowledge Graph)   │
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

### Detailed RAG Type Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG Architecture Comparison                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Classic RAG                    Agentic RAG                                 │
│  ┌─────────┐                    ┌─────────┐                                 │
│  │  Query  │                    │  Query  │                                 │
│  └────┬────┘                    └────┬────┘                                 │
│       │                              │                                      │
│       ▼                              ▼                                      │
│  ┌─────────┐                   ┌───────────┐                                │
│  │Retrieve │                   │  Agent    │                                │
│  │  Once   │                   │  Planner  │                                │
│  └────┬────┘                   └─────┬─────┘                                │
│       │                              │                                      │
│       ▼                              ▼                                      │
│  ┌─────────┐                   ┌───────────┐  ┌─────────┐                   │
│  │Generate │                   │  Tool     │◄─┤Retrieve │                   │
│  │  Once   │                   │  Selector │  │Multiple │                   │
│  └─────────┘                   └───────────┘  └────┬────┘                   │
│                                                    │                        │
│                                                    ▼                        │
│                                               ┌─────────┐                   │
│                                               │ Evaluate│                   │
│                                               │  Result │                   │
│                                               └────┬────┘                   │
│                                                    │                        │
│                                                    ▼                        │
│                                               ┌─────────┐                   │
│                                               │ Generate│                   │
│                                               │  Final  │                   │
│                                               └─────────┘                   │
│                                                                             │
│  KG-RAG                         Multimodal RAG                              │
│  ┌─────────┐                    ┌─────────┐                                 │
│  │  Query  │                    │  Query  │                                 │
│  └────┬────┘                    └────┬────┘                                 │
│       │                              │                                      │
│       ▼                              ▼                                      │
│  ┌─────────┐                  ┌───────────┐                                 │
│  │ Query   │                  │  Modality │                                 │
│  │  KG     │                  │  Detector │                                 │
│  └────┬────┘                  └─────┬─────┘                                 │
│       │                             │                                       │
│       ▼                             ▼                                       │
│  ┌─────────┐                  ┌───────────┐                                 │
│  │ Traverse│                  │ Retrieve  │                                 │
│  │ Graph   │                  │  by Type  │                                 │
│  └────┬────┘                  └─────┬─────┘                                 │
│       │                             │                                       │
│       ▼                             ▼                                       │
│  ┌──────────┐                 ┌───────────┐                                 │
│  │ Generate │                 │ Generate  │                                 │
│  │ w/Context│                 │  Response │                                 │
│  └──────────┘                 └───────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### When to Use Each Type

| Type | Complexity | Latency | Best For | Avoid When |
|------|------------|---------|----------|------------|
| **Classic RAG** | Low | Fast | Simple Q&A, chatbots | Complex multi-hop queries |
| **KG-RAG** | Medium | Medium | Relationships, reasoning | Simple document search |
| **Agentic RAG** | High | Variable | Complex workflows | Simple, fast responses needed |
| **Multimodal RAG** | High | Medium-High | Media analysis | Text-only use cases |
| **Advanced RAG** | Medium | Medium | Specialized needs | Basic Q&A |

## Key Components

### 1. Document Processing

The first step is loading and processing documents from various sources:

| Source Type | Tools | Challenges |
|-------------|-------|------------|
| PDFs | PyPDF, LangChain loaders | Layout preservation, tables |
| Web pages | BeautifulSoup, Scrapy | Dynamic content, JavaScript |
| Databases | SQL connectors | Schema mapping |
| APIs | Custom connectors | Rate limits, pagination |
| File systems | LangChain loaders | Large files, encoding |

```python
# Document Loading Example
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, WebBaseLoader, 
    Docx2txtLoader, ArxivLoader
)

# Load different document types
pdf_loader = PyPDFLoader("document.pdf")
web_loader = WebBaseLoader("https://example.com/docs")

# Process documents
pdf_docs = pdf_loader.load()
web_docs = web_loader.load()
```

### 2. Text Chunking

Splitting documents into manageable pieces for embedding:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Fixed Size** | Character/word count | Simple, fast | May break context |
| **Recursive** | Hierarchical separators | Preserves paragraphs | Complex tuning |
| **Semantic** | NLP-based splitting | Meaningful chunks | Slower |
| **Sentence** | Sentence boundaries | Natural splits | Variable sizes |

```python
# Chunking Strategies
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersRemoteTokenizer
)

# Recursive - recommended for most cases
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Token-based - good for LLMs
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = recursive_splitter.split_documents(documents)
```

### 3. Embedding Generation

Converting text into vector representations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Embedding Process                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   "RAG combines retrieval with generation"                  │
│           │                                                 │
│           ▼                                                 │
│   ┌───────────────┐     ┌─────────────────────────────┐     │
│   │ Text Input    │ ──► │  Embedding Model            │     │
│   │ (tokenized)   │     │  • OpenAI text-embedding-3  │     │
│   └───────────────┘     │  • Cohere embed-multilingual│     │
│                         │  • BGE-base-en              │     │
│                         │  • Ollama nomic-embed-text  │     │
│                         └──────────────┬──────────────┘     │
│                                        │                    │
│                                        ▼                    │
│                         ┌─────────────────────────────┐     │
│                         │  Dense Vector [0.1, -0.3,   │     │
│                         │          0.5, 0.8, ...]     │     │
│                         └─────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Model | Dimensions | API | Quality | Cost |
|-------|------------|-----|---------|------|
| OpenAI text-embedding-3-small | 1536 | ✅ | High | Low |
| OpenAI text-embedding-3-large | 3072 | ✅ | Highest | Medium |
| Cohere embed-multilingual | 1024 | ✅ | High | Medium |
| BGE-base-en | 768 | ❌ (local) | High | Free |
| Nomic embed-text | 768 | ❌ (local) | Good | Free |

### 4. Vector Database

Storing and indexing embeddings for fast retrieval:

| Database | Type | Cloud | Open Source | Best For |
|----------|------|-------|-------------|----------|
| **Pinecone** | Managed | ✅ | ❌ | Enterprise, managed |
| **Weaviate** | Managed/Self-hosted | ✅ | ✅ | Features, flexibility |
| **Chroma** | Local/Server | ❌ | ✅ | Prototyping, small scale |
| **Milvus** | Self-hosted | ✅ | ✅ | Large-scale, enterprise |
| **Qdrant** | Self-hosted | ✅ | ✅ | Performance, self-hosted |
| **pgvector** | Extension | ✅ | ✅ | Existing Postgres users |

```python
# Vector Store Examples
from langchain_community.vectorstores import Chroma, Pinecone, Weaviate
from langchain_ollama import OllamaEmbeddings

# Local (free) - Chroma with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Cloud - Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("rag-index")
vectorstore = Pinecone.from_documents(..., embedding=OpenAIEmbeddings())
```

### 5. Retrieval Strategies

How to find the most relevant documents:

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Semantic (Dense)** | Vector similarity search | Conceptual queries |
| **Keyword (BM25)** | Term frequency ranking | Exact matches, codes |
| **Hybrid** | Combines both | Mixed queries |
| **Multi-query** | Multiple search variations | Ambiguous queries |
| **Reranking** | Second-stage scoring | Precision-critical |

```python
# Retrieval Strategy Examples

# 1. Simple semantic search
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 2. Hybrid search (semantic + keyword)
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_texts(texts)
ensemble = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# 3. Retrieval with reranking
from langchain_community.cross_encoders import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### 6. Generation

Combining retrieved context with LLM generation:

```
┌─────────────────────────────────────────────────────────────┐
│                   Generation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐    ┌──────────────────────────────┐      │
│   │ User Query   │    │ Retrieved Documents (top-k)  │      │
│   └──────┬───────┘    └──────────────────────────────┘      │
│          │                         │                        │
│          │    ┌────────────────────┴────────────────────┐   │
│          │    │         Context Assembly                │   │
│          │    │  "Based on these documents:             │   │
│          │    │   [Doc 1]: ...                          │   │
│          │    │   [Doc 2]: ...                          │   │
│          │    │                                         │   │
│          │    │   Question: {query}"                    │   │
│          │    └────────────────────┬────────────────────┘   │
│          │                         │                        │
│          │                         ▼                        │
│          │              ┌─────────────────────┐             │
│          └─────────────►│        LLM          │             │
│                         │  (GPT-4, Claude,    │             │
│                         │   Llama, etc.)      │             │
│                         └──────────┬──────────┘             │
│                                    │                        │
│                                    ▼                        │
│                         ┌─────────────────────┐             │
│                         │  Generated Response │             │
│                         │  + Citations        │             │
│                         └─────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# Generation with LangChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# Define prompt template
prompt = PromptTemplate.from_template("""Use the following context 
to answer the question. If you don't know, say so.

Context:
{context}

Question: {question}

Answer:""")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"),
    chain_type="stuff",  # stuff, map_reduce, refine
    retriever=retriever,
    return_source_documents=True
)

# Execute
result = qa_chain.invoke("What is RAG?")
print(result["result"])
```

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

## RAG vs Alternatives

### Comparison with Fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAG vs Fine-tuning vs Pure LLM                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Pure LLM                    RAG                          Fine-tuning       │
│  ┌─────────┐               ┌─────────┐                   ┌─────────┐        │
│  │ Fast    │               │ Medium  │                   │ Slow    │        │
│  │ Training│               │ Latency │                   │ Training│        │
│  └────┬────┘               └────┬────┘                   └────┬────┘        │
│       │                         │                             │             │
│       ▼                         ▼                             ▼             │
│  ┌─────────┐               ┌─────────┐                   ┌─────────┐        │
│  │ Static  │               │ Dynamic │                   │ Static  │        │
│  │ Knowl-  │               │ Knowl-  │                   │ Knowl-  │        │
│  │ edge    │               │ edge    │                   │ edge    │        │
│  │         │               │ (live)  │                   │         │        │
│  └────┬────┘               └────┬────┘                   └────┬────┘        │
│       │                         │                             │             │
│       ▼                         ▼                             ▼             │
│  ┌─────────┐               ┌─────────┐                   ┌─────────┐        │
│  │ High    │               │ Low     │                   │ Medium  │        │
│  │ Hallu-  │               │ Hallu-  │                   │ Hallu-  │        │
│  │ cination│               │ cination│                   │ cination│        │
│  └─────────┘               └─────────┘                   └─────────┘        │
│                                                                             │
│  Best for:                 Best for:                      Best for:         │
│  • General chat            • Dynamic data                 • Specific style  │
│  • Creative writing        • Domain knowledge             • Task-specific   │
│  • Common knowledge        • Up-to-date info              • Repeated tasks  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Decision Framework

```
What is your primary need?
        │
        ▼
┌───────────────────────────┐
│ Need dynamic/updated data?│──Yes──► RAG recommended
└────────────┬──────────────┘
             │No
             ▼
┌───────────────────────────┐
│ Need specific domain      │──Yes──► Consider RAG vs Fine-tune
│ expertise?                │        (RAG: general domain, Fine-tune: 
└────────────┬──────────────┘             specific patterns)
             │No
             ▼
┌───────────────────────────┐
│ Need consistent style/    │──Yes──► Fine-tuning
│ tone?                     │
└────────────┬──────────────┘
             │No
             ▼
        Pure LLM likely sufficient
```

### When to Combine RAG + Fine-tuning

| Scenario | Approach |
|----------|----------|
| Domain-specific Q&A + specific style | Fine-tune base model + RAG |
| Evolving knowledge + unique terminology | Fine-tune embeddings + RAG |
| Complex reasoning + up-to-date data | Fine-tune reasoning model + RAG |

## Quick Decision Guide

| Your Situation | Recommendation |
|---------------|----------------|
| FAQ chatbot | Classic RAG |
| Code assistant | RAG + code-specific chunking |
| Research tool | Agentic RAG |
| Medical/legal research | KG-RAG |
| Media analysis | Multimodal RAG |
| Customer support | Agentic RAG |
| Internal docs | Classic RAG |
| Scientific literature | KG-RAG |

## Quick Example

```python
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

# Create test documents (replace with your actual documents)
documents = [
    Document(page_content="RAG enhances LLMs by retrieving relevant information from external knowledge bases.")
]

# Create vector store from documents (using Ollama - free, local)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

# Create QA chain (using Ollama - free, local)
qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"),
    retriever=vectorstore.as_retriever()
)

# Query
result = qa.invoke("What are the key benefits of RAG?")
print(result['result'])  # Print the answer
```

## Next Steps

- [Why RAG?](why-rag.md) - Deep dive into the motivation
- [Classic RAG Implementation](../2-architectures/classic-rag.md) - Build your first RAG
- [Technical Deep Dive](../3-technical/retrieval-systems.md) - Understand the components

---

*Previous: [Evolution of RAG](evolution-of-rag.md)*

*Next: [Why RAG](why-rag.md)*
