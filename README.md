# RAG Architecture Learning Resource

A comprehensive learning resource for understanding Retrieval-Augmented Generation (RAG) architectures, from fundamentals to advanced implementations.

## 📚 Learning Paths

This repository is organized into multiple learning paths suitable for different expertise levels:

### Beginner - Getting Started
- [What is RAG?](docs/1-theory/what-is-rag.md)
- [Why RAG?](docs/1-theory/why-rag.md)
- [Classic RAG Implementation](docs/2-architectures/classic-rag.md)

### Intermediate - Applied RAG
- [Knowledge Graph RAG](docs/2-architectures/kg-rag.md)
- [Agentic RAG](docs/2-architectures/agentic-rag.md)
- [Technical Deep Dives](docs/3-technical/)
  - [Query Rewriting](docs/3-technical/query-rewriting.md)
  - [Reranking](docs/3-technical/reranking.md)
  - [Sentence Window Retrieval](docs/3-technical/sentence-window-retrieval.md)
- [Best Practices](docs/4-best-practices/)
  - [Observability](docs/4-best-practices/observability.md)
  - [Caching](docs/4-best-practices/caching.md)

### Advanced - Research Topics
- [Multimodal RAG](docs/2-architectures/multimodal-rag.md)
- [Advanced Patterns](docs/2-architectures/advanced-patterns.md)
- [Research Directions](docs/5-pros-cons/research-directions.md)

## 📖 Content Overview

| Section | Description |
|---------|-------------|
| [Theory](docs/1-theory/) | Foundational concepts, evolution, and theory behind RAG |
| [Architectures](docs/2-architectures/) | Deep dives into Classic RAG, KG-RAG, Agentic RAG, Multimodal RAG |
| [Technical](docs/3-technical/) | Embeddings, vector databases, retrieval systems, reranking, query rewriting, evaluation, providers |
| [Best Practices](docs/4-best-practices/) | Production-ready patterns, optimization, scaling, security, observability, caching |
| [Comparison](docs/5-pros-cons/) | Pros/cons matrix, use case recommendations |

## 🧑‍💻 Interactive Notebooks

| Notebook | Description |
|----------|-------------|
| [01-classic-rag-implementation.ipynb](notebooks/01-classic-rag-implementation.ipynb) | Build your first RAG pipeline with LangChain |
| [02-kg-rag-implementation.ipynb](notebooks/02-kg-rag-implementation.ipynb) | Implement Knowledge Graph enhanced RAG |
| [03-agentic-rag-implementation.ipynb](notebooks/03-agentic-rag-implementation.ipynb) | Implement autonomous agents with LangGraph |
| [04-evaluation-workshop.ipynb](notebooks/04-evaluation-workshop.ipynb) | Measure RAG performance with RAGAS |
| [05-production-deployment.ipynb](notebooks/05-production-deployment.ipynb) | Deploy RAG to production |
| [06-advanced-retrieval-techniques.ipynb](notebooks/06-advanced-retrieval-techniques.ipynb) | Query rewriting, HyDE, and reranking |

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama** (recommended for local/privacy-friendly usage):
   - Download from: https://ollama.ai
   - Run: `ollama pull llama3.2`
   - Run: `ollama pull nomic-embed-text`

### Option 1: Using RAGProvider (Recommended)

```python
from docs._technical.providers import RAGProvider

# Using Ollama (local, free, privacy-friendly)
rag = RAGProvider(provider="ollama")
rag.add_documents([
    "RAG stands for Retrieval-Augmented Generation...",
    "RAG helps reduce hallucinations..."
])
result = rag.query("What is RAG?")

# Or using OpenAI (cloud API)
rag = RAGProvider(provider="openai")
rag.add_documents(documents)
result = rag.query("What is RAG?")

# Same interface regardless of provider!
```

### Option 2: Manual Setup

```python
# Ollama Setup (local, free) - RECOMMENDED
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents, embeddings)

# OpenAI Setup (cloud API)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbedding(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)
```

## 📁 Project Structure

```
RAG_Architecture/
├── docs/
│   ├── 1-theory/            # Foundational RAG concepts
│   ├── 2-architectures/      # RAG architecture patterns
│   ├── 3-technical/         # Technical deep dives
│   │   ├── query-rewriting.md
│   │   ├── reranking.md
│   │   ├── sentence-window-retrieval.md
│   │   └── ...
│   ├── 4-best-practices/   # Production best practices
│   │   ├── observability.md
│   │   ├── caching.md
│   │   └── ...
│   ├── 5-pros-cons/         # Comparisons and research
│   └── _technical/          # Provider implementations
│       └── providers.py     # RAGProvider class
├── notebooks/                # Interactive Jupyter notebooks
│   ├── 01-classic-rag-implementation.ipynb
│   ├── 06-advanced-retrieval-techniques.ipynb
│   └── ...
├── requirements.txt         # Python dependencies
└── README.md
```

## 🔄 Architecture Comparison

| Feature | Classic RAG | KG-RAG | Agentic RAG | Multimodal RAG |
|---------|-------------|--------|-------------|----------------|
| **Complexity** | Low | Medium | High | High |
| **Use Case** | Q&A, Docs | Relationships | Complex workflows | Video/Audio/Images |
| **Latency** | Low | Medium | Variable | Medium-High |
| **Reasoning** | Limited | Graph-based | Multi-step | Cross-modal |
| **Maintenance** | Easy | Moderate | Complex | Complex |

See [detailed comparison](docs/5-pros-cons/comparison-matrix.md)

## 📊 Evaluation Metrics

- **Retrieval**: Precision@K, Recall@K, MRR, Hit Rate
- **Generation**: BLEU, ROUGE, LLM-as-Judge
- **RAG-Specific**: RAGAS, ARES, TruLens

[Learn more about evaluation](docs/3-technical/evaluation-metrics.md)

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

## 📄 License

MIT License - feel free to use this material for learning and teaching.

## 🙏 Acknowledgments

This resource is based on the latest research in Retrieval-Augmented Generation, including:

- A Systematic Literature Review of RAG (arXiv:2508.06401)
- Agentic RAG Survey (arXiv:2501.09136)
- Comprehensive RAG Survey (arXiv:2506.00054)
- Multimodal RAG Survey (arXiv:2502.08826)
- HyDE: Hypothetical Document Embeddings (arXiv:2212.10496)
- Self-RAG (arXiv:2410.13496)

### Additional Resources

| Topic | Resource |
|-------|----------|
| HyDE | [GitHub](https://github.com/texttron/hyde) |
| Advanced RAG | [Weaviate Blog](https://weaviate.io/blog/advanced-rag) |
| Langfuse | [RAG Observability](https://langfuse.com/blog/2025-10-28-rag-observability-and-evals) |
| Query Rewriting | [DEV Community](https://dev.to/rogiia/build-an-advanced-rag-app-query-rewriting-h3p) |

---

*Last updated: March 2026*
