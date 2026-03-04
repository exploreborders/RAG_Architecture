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
- [Best Practices](docs/4-best-practices/)

### Advanced - Research Topics
- [Multimodal RAG](docs/2-architectures/multimodal-rag.md)
- [Advanced Patterns](docs/2-architectures/advanced-patterns.md)
- [Research Directions](docs/5-pros-cons/research-directions.md)

## 📖 Content Overview

| Section | Description |
|---------|-------------|
| [Theory](docs/1-theory/) | Foundational concepts, evolution, and theory behind RAG |
| [Architectures](docs/2-architectures/) | Deep dives into Classic RAG, KG-RAG, Agentic RAG, Multimodal RAG |
| [Technical](docs/3-technical/) | Embeddings, vector databases, retrieval systems, evaluation, **providers (OpenAI/Ollama)** |
| [Best Practices](docs/4-best-practices/) | Production-ready patterns, optimization, scaling |
| [Comparison](docs/5-pros-cons/) | Pros/cons matrix, use case recommendations |

## 🧑‍💻 Interactive Notebooks

| Notebook | Description |
|----------|-------------|
| [01-classic-rag-implementation.ipynb](notebooks/01-classic-rag-implementation.ipynb) | Build your first RAG pipeline with LangChain |
| [03-agentic-rag-implementation.ipynb](notebooks/03-agentic-rag-implementation.ipynb) | Implement autonomous agents with LangGraph |
| [04-evaluation-workshop.ipynb](notebooks/04-evaluation-workshop.ipynb) | Measure RAG performance with RAGAS |

## 🎯 Quick Start

```python
# Using Ollama (local, free, privacy-friendly) - RECOMMENDED
# First, install Ollama: https://ollama.ai
# Then run: ollama pull llama3.2
# And: ollama pull nomic-embed-text

from docs._technical.providers import RAGProvider

rag = RAGProvider(provider="ollama")  # Default!
rag.add_documents(documents)
result = rag.query("What is RAG?")

# OR Using OpenAI (cloud API)
rag = RAGProvider(provider="openai")
rag.add_documents(documents)
result = rag.query("What is RAG?")

# Same interface regardless of provider!
```

### Manual Setup (without wrapper)

```python
# Ollama Setup (local, free) - RECOMMENDED
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents, embeddings)

# OpenAI Setup (cloud API)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)
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

Contributions are welcome! Please read our contributing guidelines first.

## 📄 License

MIT License - feel free to use this material for learning and teaching.

## 🙏 Acknowledgments

This resource is based on the latest research in Retrieval-Augmented Generation, including:

- A Systematic Literature Review of RAG (arXiv:2508.06401)
- Agentic RAG Survey (arXiv:2501.09136)
- Comprehensive RAG Survey (arXiv:2506.00054)
- Multimodal RAG Survey (arXiv:2502.08826)

---

*Last updated: March 2026*
