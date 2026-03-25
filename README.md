# RAG Architecture Learning Resource

A comprehensive learning resource for understanding Retrieval-Augmented Generation (RAG) architectures, from fundamentals to advanced implementations.

## 📚 Learning Paths

This repository is organized into multiple learning paths suitable for different expertise levels:

### 1-Theory - Foundational Concepts
- [What is RAG?](docs/1-theory/what-is-rag.md) - Core concepts and fundamentals
- [Why RAG?](docs/1-theory/why-rag.md) - Benefits and use cases
- [Evolution of RAG](docs/1-theory/evolution-of-rag.md) - Historical development

### 2-Architectures - Implementation Patterns
- [Classic RAG](docs/2-architectures/classic-rag.md) - Basic retrieve-then-generate
- [Knowledge Graph RAG](docs/2-architectures/kg-rag.md) - Graph-enhanced retrieval
- [Agentic RAG](docs/2-architectures/agentic-rag.md) - Autonomous agent workflows
- [Multimodal RAG](docs/2-architectures/multimodal-rag.md) - Images, video, audio
- [Advanced Patterns](docs/2-architectures/advanced-patterns.md) - State-of-the-art techniques

### 3-Technical - Deep Dives
- [Embeddings Strategies](docs/3-technical/embeddings-strategies.md) - Vector representations
- [Vector Databases](docs/3-technical/vector-databases.md) - Storage and retrieval
- [Retrieval Systems](docs/3-technical/retrieval-systems.md) - Search optimization
- [Reranking](docs/3-technical/reranking.md) - Result refinement
- [Query Rewriting](docs/3-technical/query-rewriting.md) - Query enhancement
- [Sentence Window Retrieval](docs/3-technical/sentence-window-retrieval.md) - Context preservation
- [Evaluation Metrics](docs/3-technical/evaluation-metrics.md) - Performance measurement
- [Providers](docs/3-technical/providers.md) - LLM & embedding providers

### 4-Best-Practices - Production Ready
- [Chunking Strategies](docs/4-best-practices/chunking-strategies.md) - Document splitting
- [Query Optimization](docs/4-best-practices/query-optimization.md) - Performance tuning
- [Production Deployment](docs/4-best-practices/production-deployment.md) - Going live
- [Production Hardening](docs/4-best-practices/production-hardening.md) - Reliability
- [Observability](docs/4-best-practices/observability.md) - Monitoring & tracing
- [Security Considerations](docs/4-best-practices/security-considerations.md) - Protection
- [Caching](docs/4-best-practices/caching.md) - Speed optimization
- [Cost Optimization](docs/4-best-practices/cost-optimization.md) - Budget management
- [Scaling Patterns](docs/4-best-practices/scaling-patterns.md) - Growth strategies

### 5-Pros-Cons - Analysis
- [Research Directions](docs/5-pros-cons/research-directions.md) - Future trends
- [Comparison Matrix](docs/5-pros-cons/comparison-matrix.md) - Architecture trade-offs

## 📖 Content Overview

| Section | Description | Files |
|---------|-------------|-------|
| [Theory](docs/1-theory/) | Foundational concepts, evolution, and theory behind RAG | 3 |
| [Architectures](docs/2-architectures/) | Classic RAG, KG-RAG, Agentic RAG, Multimodal RAG | 5 |
| [Technical](docs/3-technical/) | Embeddings, vector databases, retrieval, reranking, evaluation | 8 |
| [Best Practices](docs/4-best-practices/) | Production patterns, optimization, scaling, security | 9 |
| [Pros/Cons](docs/5-pros-cons/) | Comparisons and research | 2 |

**Total: 27 documentation files**

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
   - Download from: https://ollama.com
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
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)
```

## 📁 Project Structure

```
RAG_Architecture/
├── docs/
│   ├── 1-theory/                    # Foundational RAG concepts
│   │   ├── what-is-rag.md
│   │   ├── why-rag.md
│   │   └── evolution-of-rag.md
│   ├── 2-architectures/             # RAG architecture patterns
│   │   ├── classic-rag.md
│   │   ├── kg-rag.md
│   │   ├── agentic-rag.md
│   │   ├── multimodal-rag.md
│   │   └── advanced-patterns.md
│   ├── 3-technical/                 # Technical deep dives
│   │   ├── embeddings-strategies.md
│   │   ├── vector-databases.md
│   │   ├── retrieval-systems.md
│   │   ├── reranking.md
│   │   ├── query-rewriting.md
│   │   ├── sentence-window-retrieval.md
│   │   ├── evaluation-metrics.md
│   │   └── providers.md
│   ├── 4-best-practices/            # Production best practices
│   │   ├── chunking-strategies.md
│   │   ├── query-optimization.md
│   │   ├── production-deployment.md
│   │   ├── production-hardening.md
│   │   ├── observability.md
│   │   ├── security-considerations.md
│   │   ├── caching.md
│   │   ├── cost-optimization.md
│   │   └── scaling-patterns.md
│   ├── 5-pros-cons/                 # Comparisons and research
│   │   ├── research-directions.md
│   │   └── comparison-matrix.md
│   └── _technical/                  # Provider implementations
│       └── providers.py              # RAGProvider class
├── notebooks/                        # Interactive Jupyter notebooks
│   ├── 01-classic-rag-implementation.ipynb
│   ├── 02-kg-rag-implementation.ipynb
│   ├── 03-agentic-rag-implementation.ipynb
│   ├── 04-evaluation-workshop.ipynb
│   ├── 05-production-deployment.ipynb
│   └── 06-advanced-retrieval-techniques.ipynb
├── requirements.txt                  # Python dependencies
├── CONTRIBUTING.md                   # Contribution guidelines
└── README.md
```

## 🔄 Architecture Comparison

| Feature | Classic RAG | KG-RAG | Agentic RAG | Multimodal RAG |
|---------|-------------|--------|-------------|---------------|
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
