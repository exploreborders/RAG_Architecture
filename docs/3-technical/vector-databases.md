# Vector Databases for RAG

## Overview

Vector databases store embeddings and enable efficient similarity search. Choosing the right vector database is crucial for RAG performance.

## How Vector Search Works

```
Vector Search Process:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                        Vector Search                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. EMBED QUERY                                                         │
│  ┌───────────────┐     ┌──────────────────┐                             │
│  │ "What is RAG?"│────►│  Embedding Model │────► [0.2, -0.1, 0.8...]    │
│  └───────────────┘     └──────────────────┘                             │
│                                                                         │
│  2. SEARCH                                                              │
│  ┌──────────────────┐     ┌──────────────────┐                          │
│  │ Query Vector     │────►│  Find Nearest    │                          │
│  │ [0.2, -0.1..]    │     │     Neighbors    │                          │
│  └──────────────────┘     └────────┬─────────┘                          │
│                                    │                                    │
│  3. RETURN RESULTS                 │                                    │
│  ┌──────────────────┐              │                                    │
│  │ Doc 1 (0.95)     │◄─────────────┘                                    │
│  │ Doc 2 (0.87)     │                                                   │
│  │ Doc 3 (0.82)     │                                                   │
│  └──────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Vector Database Comparison

| Database | Type | Scalability | Cloud Native | Best For |
|----------|------|-------------|--------------|----------|
| **Pinecone** | Managed | Excellent | Yes | Production, Enterprise |
| **Weaviate** | Open Source | Great | Yes | Flexibility, Graph |
| **Chroma** | Open Source | Good | No | Prototyping, Local |
| **Milvus** | Open Source | Excellent | Yes | Scale, Production |
| **pgvector** | Extension | Good | Yes | Existing Postgres |
| **Qdrant** | Open Source | Great | Yes | Performance |
| **Faiss** | Library | Excellent | No | Research, Offline |

## 1. Chroma

### Best For: Prototyping, Local Development

```python
"""
Chroma DB Setup and Usage
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=[
        Document(page_content="RAG is...", metadata={"source": "doc1"}),
        Document(page_content="Vector databases...", metadata={"source": "doc2"}),
    ],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)

# With score
results_with_scores = vectorstore.similarity_search_with_score("What is RAG?", k=3)

# Delete
vectorstore.delete_collection()
```

### Configuration Options

```python
from chromadb.config import Settings

# Custom configuration
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False,
    allow_reset=True
))

# Or with persistence
vectorstore = Chroma(
    client=client,
    embedding_function=embeddings,
    collection_name="my_rag"
)
```

## 2. Pinecone

### Best For: Production, Enterprise Scale

```python
"""
Pinecone Setup and Usage
"""

from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_ollama import OllamaEmbeddings
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index (if needed)
if "my-rag-index" not in pc.list_indexes().names():
    pc.create_index(
        name="my-rag-index",
        dimension=1536,  # OpenAI ada-002 dimension
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-west-2"
            }
        }
    )

# Connect
index = pc.Index("my-rag-index")

# Create vector store
vectorstore = LangChainPinecone.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    index_name="my-rag-index"
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)

# With scores
results = vectorstore.similarity_search_with_score("What is RAG?", k=3)

# Delete by ID
vectorstore.delete(ids=["doc_id1", "doc_id2"])

# Delete all
vectorstore.delete(delete_all=True)
```

### Upsert and Manage

```python
"""
Pinecone: Advanced Operations
"""

from pinecone import ServerlessSpec

# Upsert with metadata
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, ...],  # embedding
            "metadata": {
                "text": "RAG is...",
                "source": "doc1",
                "date": "2024-01-01"
            }
        }
    ],
    namespace="my-namespace"
)

# Query with filtering
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filter={
        "source": {"$eq": "doc1"},
        "date": {"$gte": "2024-01-01"}
    },
    include_metadata=True
)

# Delete namespace
index.delete_namespace("my-namespace")
```

## 3. Weaviate

### Best For: Graph Features, Flexibility

```python
"""
Weaviate Setup and Usage
"""

import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_ollama import OllamaEmbeddings

# Connect to Weaviate
client = weaviate.Client(
    url="https://my-weaviate.weaviate.io",
    auth_client_secret=weaviate.AuthApiKey(api_key="YOUR-API-KEY"),
    additional_headers={
        "X-OpenAI-Api-Key": "YOUR-OPENAI-KEY"
    }
)

# Create vector store
vectorstore = Weaviate.from_documents(
    client=client,
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    index_name="MyRAG",
    text_key="text",
    by_json_path=["content"]  # Path to text in schema
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)

# With filters
results = vectorstore.similarity_search(
    "What is RAG?",
    k=3,
    filter={"path": ["source"], "operator": "Equal", "valueString": "doc1"}
)
```

### Graph Queries

```python
"""
Weaviate Graph Queries
"""

# Near text search
results = client.query.get(
    "Article",
    ["title", "content", "author { name }"]
).with_near_text({
    "concepts": ["artificial intelligence"]
}).with_limit(10).do()

# Hybrid search (keyword + semantic)
results = client.query.get(
    "Article",
    ["title", "content"]
).with_hybrid(
    query="RAG",
    alpha=0.5  # 0 = keyword, 1 = semantic
).with_limit(10).do()
```

## 4. Milvus

### Best For: Large Scale, Production

```python
"""
Milvus Setup and Usage
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain_community.vectorstores import Milvus
from langchain_ollama import OllamaEmbeddings

# Connect
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
]
schema = CollectionSchema(fields=fields, description="RAG collection")
collection = Collection(name="rag_docs", schema=schema)

# Create index
index_params = {
    "metric_type": "IP",  # Inner Product
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

# Load into LangChain
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="rag_docs",
    connection_args={"host": "localhost", "port": "19530"}
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)
```

## 5. pgvector

### Best For: Existing Postgres Infrastructure

```python
"""
pgvector Setup and Usage
"""

from pgvector.psycopg2 import register_vector
from langchain_community.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
import psycopg2

# Create extension (run once)
# CREATE EXTENSION vector;
# ALTER TABLE langchain_pg_embedding ADD COLUMN embedding vector(1536);

# Connect with pgvector
COLLECTION_NAME = "rag_documents"
CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/vector_db"

# Create vector store
vectorstore = PGVector.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)

# With custom query (filtering)
results = vectorstore.similarity_search(
    "What is RAG?",
    k=3,
    filter={"source": "doc1"}
)
```

### SQL Examples

```python
"""
pgvector: Direct SQL Operations
"""

# Find similar embeddings
query = """
SELECT id, content, 
       embedding <=> %s AS distance
FROM documents
ORDER BY embedding <=> %s
LIMIT 5;
"""

# Filtered search
filtered_query = """
SELECT id, content
FROM documents
WHERE source = %s
ORDER BY embedding <=> %s
LIMIT 5;
"""
```

## 6. Qdrant

### Best For: Performance, Ease of Use

```python
"""
Qdrant Setup and Usage
"""

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings

# Connect
client = QdrantClient(host="localhost", port=6333)

# Create vector store
vectorstore = Qdrant.from_documents(
    client=client,
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="rag_docs"
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)

# With scores
results = vectorstore.similarity_search_with_score("What is RAG?", k=3)

# With payload filter
results = vectorstore.similarity_search(
    "What is RAG?",
    k=3,
    filter="source == 'doc1'"
)
```

## 7. Faiss (Facebook AI Similarity Search)

### Best For: Research, Offline Processing, Maximum Control

Faiss is a library for efficient similarity search, not a full database. It's excellent when you need maximum control or are working offline.

```python
"""
Faiss Setup and Usage
"""

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Sample documents
documents = [
    Document(page_content="RAG is Retrieval-Augmented Generation", metadata={"source": "doc1"}),
    Document(page_content="Vector databases store embeddings", metadata={"source": "doc2"}),
]

# Get embedding dimension
sample_embedding = embeddings.embed_query("test")
dimension = len(sample_embedding)

# Create Faiss index (IP = Inner Product, use L2 for Euclidean)
index = faiss.IndexFlatIP(dimension)

# Create FAISS vector store
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=2)

# With scores
results_with_scores = vectorstore.similarity_search_with_score("What is RAG?", k=2)
```

### Advanced Faiss Operations

```python
"""
Faiss: Advanced Index Types
"""

# HNSW index (faster search, more memory)
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 = number of connections

# IVF index (faster search on large datasets)
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatIP(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Train on your data (required for IVF)
# embeddings_matrix = ... (your embeddings)
# index_ivf.train(embeddings_matrix)

# Add and search
# index_ivf.add(embeddings_matrix)
# D, I = index_ivf.search(query_embedding, k=10)
```

## Indexing Strategies

### HNSW vs IVF

| Index Type | Build Time | Query Speed | Memory | Best For         |
|------------|------------|-------------|--------|------------------|
| **HNSW**   | Medium     | Very Fast   | High   | Latency-critical |
| **IVF**    | Fast       | Fast        | Medium | Balanced         |
| **Flat**   | Very Fast  | Slow        | High   | Small datasets   |

```python
"""
Choosing Index Strategy
"""

# Pinecone (serverless)
index = pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",  # or "euclidean", "dotproduct"
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)

# Qdrant (custom)
client.create_collection(
    collection_name="my-collection",
    vectors_config=VectorParams(
        size=1536,
        distance="Cosine",
        hnsw_config=HnswConfig(
            m=16,  # Connections per node
            ef_construct=200  # Search width
        )
    )
)
```

## Performance Optimization Tips

```python
"""
Performance Best Practices
"""

# 1. Batch embeddings
def batch_embed(texts: list, batch_size: int = 100):
    """Batch process for efficiency."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        yield embeddings.embed_documents(batch)

# 2. Async operations
import asyncio

async def async_search(vectorstore, queries: list):
    """Concurrent searches."""
    tasks = [vectorstore.asimilarity_search(q) for q in queries]
    return await asyncio.gather(*tasks)

# 3. Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query: str):
    """Cache frequent queries."""
    return vectorstore.similarity_search(query)

# 4. Pagination for large results
def paginated_search(query: str, limit: int = 100, offset: int = 0):
    """Handle large result sets."""
    return vectorstore.similarity_search(
        query,
        k=limit,
        filter={"offset": offset}  # Implementation depends on DB
    )
```

## Decision Guide

```
Which Vector Database?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start
  │
  ▼
Need a full database or just a search library?
  │
  ├─ Just search library ──► Faiss (research/offline)
  │                            │
  │                            │ Need more features?
  │                            ├─ No ─► Stay with Faiss
  │                            └─ Yes ─► Switch to database below
  │
  └─ Full database ──► Continue below
                       │
                       ▼
How large is your dataset?
  │
  ├─ < 100K vectors ──► Chroma (prototyping)
  │                            │
  │                            │ Need production?
  │                            ├─ No ─► Stay with Chroma
  │                            └─ Yes ─► Pinecone / Qdrant
  │
  ├─ 100K - 10M ──► Pinecone / Qdrant / Milvus
  │
  └─ > 10M ──► Milvus / Pinecone (enterprise)
       │
       ▼
Do you have existing infrastructure?
  │
  ├─ Yes (Postgres) ──► pgvector
  ├─ Yes (Cloud) ──► Pinecone / Weaviate
  │
  └─ No ──► Any of above
       │
       ▼
What's your priority?
  │
  ├─ Ease of use ──► Chroma / Qdrant
  ├─ Performance ──► Milvus / Qdrant
  ├─ Features ──► Weaviate
  ├─ Managed/Cloud ──► Pinecone
  └─ Cost ──► Chroma / pgvector
```

---

## References

### Official Documentation

| Database | Documentation |
|----------|---------------|
| [Chroma](https://docs.trychroma.com/) | ChromaDB docs |
| [Pinecone](https://docs.pinecone.io/) | Pinecone docs |
| [Qdrant](https://qdrant.tech/documentation/) | Qdrant docs |
| [Weaviate](https://weaviate.io/developers/weaviate) | Weaviate docs |
| [Milvus](https://milvus.io/docs) | Milvus docs |
| [pgvector](https://github.com/pgvector/pgvector) | pgvector GitHub |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [Vector Database Comparison](https://www.pinecone.io/learn/vector-database-comparison) | DB selection guide |
| [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search) | Hybrid search guide |
| [Pinecone Serverless](https://www.pinecone.io/learn/serverless) | Serverless architecture |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [chroma](https://github.com/chroma-core/chroma) | ChromaDB |
| [milvus](https://github.com/milvus-io/milvus) | Milvus |
| [qdrant](https://github.com/qdrant/qdrant) | Qdrant |

---

*Next: [Production Deployment](../4-best-practices/production-deployment.md)*
