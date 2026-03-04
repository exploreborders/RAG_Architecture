# LLM and Embedding Providers

## Overview

This guide covers different LLM and embedding providers, with a focus on using OpenAI and Ollama (local models). We provide a unified wrapper that lets you switch between providers easily.

## When to Use What

```
Provider Selection Guide:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Need privacy/offline?
│
├─ Yes ──► Ollama (local) ◄── DEFAULT
│
└─ No ──► Quality vs Cost?
            │
            ├─ Best quality, budget OK ──► OpenAI GPT-4 / Claude
            │
            ├─ Good quality, budget tight ──► Ollama (free!)
            │
            └─ Development/testing ──► Ollama (free)
```

## Provider Comparison

| Provider | Type | Cost | Quality | Privacy | Setup |
|----------|------|------|---------|---------|-------|
| **Ollama** | Local | Free | Good-Very Good | 100% local | Medium |
| **OpenAI** | Cloud API | Pay-per-use | Excellent | Data leaves local | Easy |
| **Anthropic** | Cloud API | Pay-per-use | Excellent | Data leaves local | Easy |
| **HuggingFace** | Cloud/Local | Free tier / Local | Good | Varies | Medium |

## Unified Wrapper Class

### Basic Wrapper

```python
"""
LLMProvider: Unified interface for OpenAI and Ollama
"""

import os
from enum import Enum
from typing import Optional, Union

class ProviderType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"

class LLMProvider:
    """
    Unified LLM and Embedding provider.
    
    Usage:
        provider = LLMProvider()  # Default: Ollama (local, free)
        provider = LLMProvider("openai")  # Use OpenAI API
    """
    
    def __init__(
        self,
        provider: str = "ollama",  # Default to Ollama!
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = ProviderType(provider.lower())
        self.llm_model = llm_model or self._default_llm()
        self.embedding_model = embedding_model or self._default_embedding()
        
        # Configuration
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize clients
        self._llm = None
        self._embeddings = None
    
    def _default_llm(self) -> str:
        """Get default LLM model for provider."""
        defaults = {
            ProviderType.OPENAI: "gpt-4o",
            ProviderType.OLLAMA: "llama3.2",
            ProviderType.ANTHROPIC: "claude-3-haiku-20240307",
        }
        return defaults.get(self.provider, "llama3.2")  # Default to llama3.2
    
    def _default_embedding(self) -> str:
        """Get default embedding model for provider."""
        defaults = {
            ProviderType.OPENAI: "text-embedding-3-small",
            ProviderType.OLLAMA: "nomic-embed-text",
        }
        return defaults.get(self.provider, "nomic-embed-text")  # Default to nomic-embed-text
    
    @property
    def llm(self):
        """Get LLM client."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    @property
    def embeddings(self):
        """Get embedding client."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_llm(self):
        """Create LLM client based on provider."""
        
        if self.provider == ProviderType.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.llm_model,
                api_key=self.api_key,
                temperature=0
            )
        
        elif self.provider == ProviderType.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.llm_model,
                base_url=self.base_url or "http://localhost:11434",
                temperature=0
            )
        
        elif self.provider == ProviderType.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_model,
                api_key=self.api_key
            )
        
        raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_embeddings(self):
        """Create embedding client based on provider."""
        
        if self.provider == ProviderType.OPENAI:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key
            )
        
        elif self.provider == ProviderType.OLLAMA:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url or "http://localhost:11434"
            )
        
        raise ValueError(f"Unknown provider: {self.provider}")
```

### Complete RAG Wrapper

```python
"""
Complete RAG wrapper with provider support
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain.chains import RetrievalQA

class RAGProvider:
    """
    Complete RAG system with multi-provider support.
    
    Usage:
        # Use Ollama (local, free) - DEFAULT
        rag = RAGProvider()  # Uses Ollama automatically
        
        # Use OpenAI (cloud)
        rag = RAGProvider(provider="openai")
        
        # Add documents
        rag.add_documents(documents)
        
        # Query
        result = rag.query("What is RAG?")
    """
    
    def __init__(
        self,
        provider: str = "ollama",  # Default to Ollama!
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_directory: str = "./vector_db"
    ):
        self.provider = LLMProvider(
            provider=provider,
            llm_model=llm_model,
            embedding_model=embedding_model
        )
        self.vector_store_type = vector_store
        self.persist_directory = persist_directory
        self._vectorstore = None
        self._qa_chain = None
    
    @property
    def vectorstore(self):
        """Get or create vector store."""
        if self._vectorstore is None:
            self._vectorstore = self._create_vectorstore()
        return self._vectorstore
    
    def _create_vectorstore(self) -> VectorStore:
        """Create vector store."""
        
        if self.vector_store_type == "chroma":
            from langchain_chroma import Chroma
            return Chroma(
                embedding_function=self.provider.embeddings,
                persist_directory=self.persist_directory
            )
        
        elif self.vector_store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            return FAISS.from_documents(
                documents=[],
                embedding=self.provider.embeddings
            )
        
        raise ValueError(f"Unknown vector store: {self.vector_store_type}")
    
    def add_documents(self, documents: list, batch_size: int = 100):
        """Add documents to the vector store."""
        
        if self._vectorstore is None:
            # Create new with documents
            from langchain_chroma import Chroma
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.provider.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Add to existing
            self._vectorstore.add_documents(documents)
    
    def create_qa_chain(self, chain_type: str = "stuff"):
        """Create QA chain."""
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self.provider.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question: str) -> dict:
        """Query the RAG system."""
        
        if self._qa_chain is None:
            self.create_qa_chain()
        
        result = self._qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ]
        }
    
    def get_model_info(self) -> dict:
        """Get information about current models."""
        
        return {
            "provider": self.provider.provider.value,
            "llm_model": self.provider.llm_model,
            "embedding_model": self.provider.embedding_model,
            "vector_store": self.vector_store_type
        }
```

## Usage Examples

### Quick Start with Ollama (Default)

```python
# Using Ollama (local, free, privacy-friendly) - DEFAULT
# First, install Ollama: https://ollama.ai
# Then run: ollama pull llama3.2
# And: ollama pull nomic-embed-text

rag = RAGProvider()  # Uses Ollama by default!
rag.add_documents(documents)
result = rag.query("What is RAG?")

print(result["answer"])
```

### Quick Start with OpenAI

```python
# Using OpenAI (requires API key)
rag = RAGProvider(provider="openai")
rag.add_documents(documents)
result = rag.query("What is RAG?")

print(result["answer"])
```

### Environment-Based Configuration

```python
"""
Use environment variable to switch providers
"""

import os

def create_rag_from_env():
    """Create RAG based on environment."""
    
    provider = os.getenv("RAG_PROVIDER", "openai")
    
    return RAGProvider(
        provider=provider,
        llm_model=os.getenv("LLM_MODEL"),
        embedding_model=os.getenv("EMBEDDING_MODEL")
    )

# .env file:
# RAG_PROVIDER=ollama
# LLM_MODEL=llama3.2
# EMBEDDING_MODEL=nomic-embed-text
```

### Switching Providers

```python
"""
Easily switch between providers
"""

# Development/Learning: Use Ollama (free)
rag_dev = RAGProvider(provider="ollama")  # Default!

# Production: Use OpenAI (if needed)
rag_production = RAGProvider(provider="openai", llm_model="gpt-4")

# Same interface!
print(rag_dev.query("What is RAG?")["answer"])
print(rag_production.query("What is RAG?")["answer"])
```

## Ollama Setup

### Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai
```

### Pull Required Models

```bash
# Pull Llama 3.2 (general purpose)
ollama pull llama3.2

# Pull for coding
ollama pull codellama

# Pull for embeddings
ollama pull nomic-embed-text

# List installed models
ollama list
```

### Start Ollama Server

```bash
# Start server (runs on port 11434 by default)
ollama serve

# Or run in background
ollama serve &
```

## Model Recommendations

### For OpenAI

| Use Case | Model | Cost | Notes |
|----------|-------|------|-------|
| Best quality | gpt-4o | High | Most capable |
| Balanced | gpt-4o-mini | Low | Fast, cheap |
| Legacy | gpt-4 | High | Still excellent |
| Embeddings | text-embedding-3-small | Low | Great quality |

### For Ollama

| Use Case | Model | RAM Needed | Notes |
|----------|-------|------------|-------|
| General | llama3.2 | 4GB | Good quality |
| Coding | codellama | 8GB | Code optimized |
| Fast | mistral | 4GB | Very fast |
| Embeddings | nomic-embed-text | 1GB | Good quality |

## Troubleshooting

### Ollama Issues

```python
# If Ollama not running
# Start: ollama serve

# Check connection
import requests
response = requests.get("http://localhost:11434/api/tags")
print(response.json())

# If wrong model
# Pull: ollama pull llama3.2
```

### OpenAI Issues

```python
# Check API key
import os
print(os.getenv("OPENAI_API_KEY"))

# Check quota
from openai import OpenAI
client = OpenAI()
# Usage will show your current usage
```

---

*Related: [Embedding Strategies](embeddings-strategies.md) • [Cost Optimization](../4-best-practices/cost-optimization.md)*
