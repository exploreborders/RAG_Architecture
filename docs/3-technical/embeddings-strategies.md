# Embeddings Strategies

## Overview

Embeddings convert text into vector representations that capture semantic meaning. Choosing the right embedding strategy is crucial for RAG quality.

## How Embeddings Work

```
Embedding Process:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Text Input                    Embedding Model                  Vector Output
┌─────────────┐             ┌─────────────────┐            ┌─────────────┐
│ "What is    │────────────►│                 │───────────►│ [0.12,      │
│  RAG?"      │             │   Embedding     │            │  -0.34,     │
│             │             │   Model         │            │   0.78,     │
│             │             │                 │            │  ...]       │
└─────────────┘             └─────────────────┘            └─────────────┘

Similar texts → Similar vectors → Near each other in vector space
```

## Embedding Models Comparison

| Model | Dimensions | Speed | Quality | Cost | Best For |
|-------|------------|-------|---------|------|----------|
| **nomic-embed-text** | 768 | Fast | Good | Free | Default - Local |
| **text-embedding-3-small** | 1536 | Fast | Good | Low | Cloud API |
| **text-embedding-3-large** | 3072 | Medium | Excellent | Medium | High quality |
| **BGE-large** | 1024 | Medium | Excellent | Free | Open source |
| **BGE-base** | 768 | Fast | Good | Free | Balanced |

## Implementation

### Prerequisites

```bash
# Install Ollama: https://ollama.ai
ollama pull nomic-embed-text
```

### Ollama Embeddings (Default - Free, Local)

```python
"""
Ollama Embeddings (Default - Free, Local)
"""

from langchain_ollama import OllamaEmbeddings

# Default model: nomic-embed-text (free!)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Create embeddings
text = "What is Retrieval Augmented Generation?"
vector = embeddings.embed_query(text)

# Batch embedding
documents = ["Doc 1", "Doc 2", "Doc 3"]
vectors = embeddings.embed_documents(documents)
```

### OpenAI Embeddings (Alternative - Cloud API)

```python
"""
OpenAI Embeddings (Alternative - Cloud API)
"""

from langchain_openai import OpenAIEmbeddings

# Latest model (text-embedding-3)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536  # Optional: reduce for cost savings
)

# Create embeddings
text = "What is Retrieval-Augmented Generation?"
vector = embeddings.embed_query(text)

```python
"""
BGE (BAAI General Embedding)
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

# BGE-large (best quality)
embeddings_large = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"}  # or "cpu"
)

# BGE-base (balanced)
embeddings_base = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"}
)
```

### Cohere Embeddings

```python
"""
Cohere Embeddings
"""

from langchain_community.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings(
    cohere_api_key="your-key",
    model="embed-english-v3.0"
)
```

## Advanced Embedding Strategies

### 1. Late Chunking

```python
"""
Late Chunking: Embed first, then chunk
"""

from transformers import AutoModel, AutoTokenizer
import torch

class LateChunkingEmbedder:
    """Embed at document level, then extract chunk embeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed_document(self, text: str, chunk_size: int = 512):
        """Embed full document with pooling."""
        
        # Tokenize full document
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        # Get all token embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            token_embeddings = outputs.last_hidden_state[0]
        
        # Split into chunks and pool
        chunks = []
        for i in range(0, len(token_embeddings), chunk_size):
            chunk_emb = token_embeddings[i:i+chunk_size]
            # Mean pooling
            pooled = chunk_emb.mean(dim=0)
            chunks.append(pooled.numpy())
        
        return chunks
```

### 2. Multi-Vector Embeddings

```python
"""
Store multiple vectors per chunk for richer representation
"""

from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings

class MultiVectorEmbedder:
    """Create multiple embeddings per document."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
    
    def embed_documents(self, documents: list) -> list:
        """Create multiple vectors per document."""
        
        embedded = []
        
        for doc in documents:
            vectors = {
                "full": self.embeddings.embed_query(doc.page_content),
                "summary": self.embeddings.embed_query(
                    self._summarize(doc.page_content)
                ),
                "keywords": self.embeddings.embed_query(
                    self._extract_keywords(doc.page_content)
                )
            }
            embedded.append({
                "document": doc,
                "vectors": vectors
            })
        
        return embedded
    
    def _summarize(self, text: str) -> str:
        # Use LLM or extractive summarization
        return text[:200] + "..."
    
    def _extract_keywords(self, text: str) -> str:
        # Extract key terms
        return text.split()[:10]
```

### 3. Sentence Window Embeddings

```python
"""
Sentence Window: Embed sentences with surrounding context
"""

class SentenceWindowEmbedder:
    """Embed sentences with surrounding context."""
    
    def __init__(self, embeddings, window_size: int = 3):
        self.embeddings = embeddings
        self.window_size = window_size
    
    def create_windows(self, sentences: list) -> list:
        """Create sentence windows."""
        
        windows = []
        
        for i, sentence in enumerate(sentences):
            # Get surrounding sentences
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)
            
            context = " ".join(sentences[start:end])
            
            windows.append({
                "window": context,
                "sentence": sentence,
                "position": i
            })
        
        return windows
    
    def embed_windows(self, windows: list) -> list:
        """Embed each window."""
        
        return [
            self.embeddings.embed_query(w["window"])
            for w in windows
        ]
```

### 4. Hierarchical Embeddings

```python
"""
Different embeddings for different granularities
"""

class HierarchicalEmbedder:
    """Create embeddings at multiple granularities."""
    
    def __init__(self):
        # Different models for different levels
        self.embeddings = {
            "chunk": OpenAIEmbeddings(model="text-embedding-3-small"),
            "summary": OpenAIEmbeddings(model="text-embedding-3-large"),
        }
    
    def embed_document(self, document: str) -> dict:
        """Embed at multiple levels."""
        
        # Split into chunks
        chunks = self._split_chunks(document)
        
        # Embed chunks
        chunk_vectors = self.embeddings["chunk"].embed_documents(chunks)
        
        # Create and embed summary
        summary = self._create_summary(document)
        summary_vector = self.embeddings["summary"].embed_query(summary)
        
        return {
            "chunks": list(zip(chunks, chunk_vectors)),
            "summary": (summary, summary_vector)
        }
    
    def _split_chunks(self, text: str) -> list:
        # Simple split
        return text.split("\n\n")
    
    def _create_summary(self, text: str) -> str:
        # Use LLM for summary
        return text[:300]
```

## Embedding Optimization

### Dimension Reduction

```python
"""
Reduce embedding dimensions for efficiency
"""

from sklearn.decomposition import PCA
import numpy as np

def reduce_dimensions(vectors, target_dim: int = 256):
    """Reduce embedding dimensions using PCA."""
    
    vectors_array = np.array(vectors)
    
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(vectors_array)
    
    print(f"Reduced from {vectors_array.shape[1]} to {target_dim}")
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    
    return reduced.tolist()
```

### Quantization

```python
"""
Quantize embeddings for storage efficiency
"""

import numpy as np

def quantize_embeddings(vectors, bits: int = 8):
    """Quantize float vectors to lower precision."""
    
    vectors_array = np.array(vectors)
    
    # Calculate range
    vmin, vmax = vectors_array.min(), vectors_array.max()
    
    # Quantize
    if bits == 8:
        quantized = ((vectors_array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    elif bits == 4:
        quantized = ((vectors_array - vmin) / (vmax - vmin) * 15).astype(np.uint8)
    
    return quantized, vmin, vmax

def dequantize_embeddings(quantized, vmin, vmax, original_shape):
    """Restore quantized vectors."""
    
    dequantized = quantized.astype(np.float32) / 255 * (vmax - vmin) + vmin
    
    return dequantized.reshape(original_shape)
```

## Choosing the Right Embedding

```
Embedding Selection Guide:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start
  │
  ▼
Budget?
  │
  ├─ Low budget ──► BGE (open source)
  │
  └─ Can pay ──► Quality needed?
                    │
                    ├─ Production ──► Cohere / OpenAI-3-small
                    │
                    └─ Best quality ──► OpenAI-3-large / BGE-large

Special needs?
  │
  ├─ Multilingual ──► BGE-multilingual / Cohere
  │
  ├─ Code ──► code-bge / OpenAI-code
  │
  └─ Long documents ──► Late chunking strategy
```

## Benchmarking Embeddings

```python
"""
Compare embedding models
"""

from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

def benchmark_embeddings(embeddings, test_pairs: list) -> dict:
    """Benchmark embedding model."""
    
    import time
    
    # Test data: (text1, text2, expected_similarity)
    test_cases = [
        ("cat", "kitten", 0.9),  # High similarity
        ("car", "banana", 0.1),  # Low similarity
    ]
    
    results = []
    
    for text1, text2, expected in test_cases:
        start = time.time()
        
        v1 = embeddings.embed_query(text1)
        v2 = embeddings.embed_query(text2)
        
        # Cosine similarity
        import numpy as np
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        results.append({
            "texts": (text1, text2),
            "similarity": sim,
            "expected": expected,
            "error": abs(sim - expected),
            "latency_ms": (time.time() - start) * 1000
        })
    
    return {
        "avg_error": sum(r["error"] for r in results) / len(results),
        "avg_latency": sum(r["latency_ms"] for r in results) / len(results)
    }
```

## Related Topics

- [Cost Optimization](../4-best-practices/cost-optimization.md) - Optimize embedding costs
- [Chunking Strategies](../4-best-practices/chunking-strategies.md) - Optimize chunk sizes for embeddings

---

*Next: [Retrieval Systems](retrieval-systems.md)*
