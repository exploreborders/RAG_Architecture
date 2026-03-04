# RAG Architecture Learning Resource

A comprehensive learning resource for understanding Retrieval-Augmented Generation (RAG) architectures, from fundamentals to advanced implementations.

## 📚 Learning Paths

This repository is organized into multiple learning paths suitable for different expertise levels:

### Beginner - Getting Started
- [What is RAG?](docs/1-theory/what-is-rag.md)
- [Why RAG?](docs/1-theory/why-rag.md)
- [Classic RAG Implementation](docs/2-architectures/classic-rag/)

### Intermediate - Applied RAG
- [Knowledge Graph RAG](docs/2-architectures/kg-rag/)
- [Agentic RAG](docs/2-architectures/agentic-rag/)
- [Technical Deep Dives](docs/3-technical/)
- [Best Practices](docs/4-best-practices/)

### Advanced - Research Topics
- [Multimodal RAG](docs/2-architectures/multimodal-rag/)
- [Advanced Patterns](docs/2-architectures/advanced-patterns/)
- [Research Directions](docs/5-pros-cons/research-directions.md)

## 📖 Content Overview

| Section | Description |
|---------|-------------|
| [Theory](docs/1-theory/) | Foundational concepts, evolution, and theory behind RAG |
| [Architectures](docs/2-architectures/) | Deep dives into Classic RAG, KG-RAG, Agentic RAG, Multimodal RAG |
| [Technical](docs/3-technical/) | Embeddings, vector databases, retrieval systems, evaluation |
| [Best Practices](docs/4-best-practices/) | Production-ready patterns, optimization, scaling |
| [Comparison](docs/5-pros-cons/) | Pros/cons matrix, use case recommendations |

## 🧑‍💻 Interactive Notebooks

| Notebook | Description |
|----------|-------------|
| [01-classic-rag-implementation.ipynb](notebooks/01-classic-rag-implementation.ipynb) | Build your first RAG pipeline |
| [02-kg-rag-implementation.ipynb](notebooks/02-kg-rag-implementation.ipynb) | Add knowledge graph reasoning |
| [03-agentic-rag-implementation.ipynb](notebooks/03-agentic-rag-implementation.ipynb) | Implement autonomous agents |
| [04-evaluation-workshop.ipynb](notebooks/04-evaluation-workshop.ipynb) | Measure RAG performance |
| [05-production-deployment.ipynb](notebooks/05-production-deployment.ipynb) | Deploy to production |

## 🎯 Quick Start

```python
# Simple RAG with LangChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. Create QA chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# 5. Ask questions
result = qa_chain.run("What is the main topic?")
print(result)
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
