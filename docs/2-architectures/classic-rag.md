# Classic RAG (Naive RAG)

## Overview

**Classic RAG** (also known as Naive RAG) is the foundational RAG architecture that introduced the retrieve-then-generate paradigm. It provides a straightforward pipeline from query to response.

## Architecture

```
Classic RAG Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        ┌─────────────────────┐
                        │   User Query        │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Embedding Model    │
                        │  (Convert query to  │
                        │   vector)           │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Vector Database   │
                        │   (Search for       │
                        │    similar chunks)  │
                        └──────────┬──────────┘
                                   │
                        ┌──────────┴──────────┐
                        │                     │
                        ▼                     ▼
                 ┌─────────────┐       ┌─────────────┐
                 │   Top-K     │       │   Prompt    │
                 │  Retrieved  │──────►│  Template   │
                 │   Chunks    │       │             │
                 └─────────────┘       └──────┬──────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │     LLM     │
                                        │  Generation │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Final     │
                                        │  Response   │
                                        └─────────────┘
```

## Components

### Prerequisites

```bash
# Install Ollama: https://ollama.ai
# Then pull required models:
ollama pull llama3.2      # LLM
ollama pull nomic-embed-text  # Embeddings
```

### 1. Document Loader
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt")
documents = loader.load()
```

### 2. Text Splitter
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = splitter.split_documents(documents)
```

### 3. Embedding Model (Ollama)
```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### 4. Vector Store
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)
```

### 5. Retriever
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 6. QA Chain (Ollama)
```python
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

## Complete Example (LangChain) - Using Ollama

```python
"""
Classic RAG Implementation with LangChain + Ollama
Prerequisites: Install Ollama and pull models:
  ollama pull llama3.2
  ollama pull nomic-embed-text
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configuration
DOC_PATH = "data/document.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# 1. Load Document
loader = TextLoader(DOC_PATH)
documents = loader.load()

# 2. Split Text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
texts = splitter.split_documents(documents)

# 3. Create Embeddings (Ollama - local, free)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Build Vector Store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# 5. Create Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

# 6. Custom Prompt
prompt_template = """Use the following context to answer the question.
If you don't know the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 7. Create QA Chain (Ollama - local, free)
llm = ChatOllama(model="llama3.2", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Query
query = "What is the main topic of the document?"
result = qa_chain({"query": query})

print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata.get('source', 'Unknown')}")
```

## Complete Example (LlamaIndex) - Using Ollama

```python
"""
Classic RAG Implementation with LlamaIndex + Ollama
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings

# Configure Settings to use Ollama
Settings.llm = Ollama(model="llama3.2")

# Use LangChain Ollama embeddings
lc_embeddings = OllamaEmbeddings(model="nomic-embed-text")
embedding = LangchainEmbedding(lc_embeddings)

# 1. Load Documents (or use sample documents)
# documents = SimpleDirectoryReader("data").load_data()
documents = [Document(text="RAG is a technique for AI information retrieval.")]

# 2. Create Index with Ollama embeddings
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embedding,
    chunk_size=1000,
    chunk_overlap=200
)

# 3. Create Query Engine (using Ollama)
query_engine = index.as_query_engine()

# 4. Query
response = query_engine.query("What is RAG?")

print(response)
print("\nSources:")
for source in response.source_nodes:
    print(f"- {source.node.text[:100]}...")
```

### Ollama Models Comparison

| Model | Size | Best For | RAM |
|-------|------|----------|-----|
| llama3.2 | 3.8GB | General purpose | ~4GB |
| mistral | 4.1GB | Fast, efficient | ~4GB |
| codellama | 3.8GB | Code tasks | ~4GB |
| llama3.2:70b | 40GB | Best quality | ~64GB |
| nomic-embed-text | 274MB | Embeddings | ~1GB |

## OpenAI vs Ollama Comparison

| Aspect | OpenAI | Ollama |
|--------|--------|--------|
| **Cost** | Pay-per-use | Free (compute only) |
| **Setup** | API key only | Install + download models |
| **Privacy** | Data leaves local | 100% local |
| **Quality** | Excellent (GPT-4) | Good (Llama 3.2) |
| **Speed** | Fast (cloud) | Depends on hardware |
| **Offline** | No | Yes |
| **Customization** | Limited | Full control |

## When to Use Each

### Use OpenAI when:
- Need highest quality responses
- Have budget for API costs
- Don't need offline operation
- Want minimal setup

### Use Ollama when:
- Privacy is critical (data stays local)
- Building internal/private tools
- Learning RAG (free)
- Need offline operation

## Pros and Cons

### ✅ Advantages

| Advantage | Description |
|-----------|-------------|
| **Simple** | Easy to understand and implement |
| **Fast** | Low latency, straightforward pipeline |
| **Reliable** | Predictable behavior |
| **Well-supported** | Extensive documentation and tools |
| **Cost-effective** | Minimal infrastructure |

### ❌ Limitations

| Limitation | Description |
|------------|-------------|
| **No multi-hop** | Cannot handle complex reasoning |
| **Single retrieval** | No iterative refinement |
| **No planning** | Linear pipeline only |
| **Fixed context** | Doesn't adapt to query complexity |
| **No memory** | No conversation history |

## When to Use Classic RAG

### ✅ Best For

- Simple Q&A over documents
- Knowledge base chatbots
- FAQ systems
- Document summarization
- Basic information retrieval

### ❌ Not Ideal For

- Multi-step reasoning
- Complex queries requiring decomposition
- Dynamic retrieval strategies
- Conversations with history
- Graph-structured data

## Performance Considerations

### Retrieval Strategies

```python
# Different retrieval strategies
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Basic semantic search
    # search_type="mmr"        # Max marginal relevance
    # search_type="similarity_threshold"  # Threshold-based
    search_kwargs={"k": 4}
)
```

### Chunk Size Trade-offs

| Chunk Size | Pros | Cons |
|------------|------|------|
| **Small (256-512)** | Precise, less noise | May lose context |
| **Medium (512-1024)** | Balanced | May truncate concepts |
| **Large (1024+)** | Full context | More noise, higher cost |

### Top-K Selection

| K Value | Use Case |
|---------|----------|
| **k=1-2** | Precise answers, small contexts |
| **k=3-5** | Balanced, general Q&A |
| **k=6-10** | Comprehensive, complex queries |

## Evaluation

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.models import OllamaModel

# Initialize with Ollama
ollama_model = OllamaModel(model="llama3.2")

# Create test cases
test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG stands for Retrieval-Augmented Generation...",
    retrieval_context=["RAG is a framework...", "It combines retrieval..."]
)

# Evaluate
faithfulness = FaithfulnessMetric(model=ollama_model)
answer_relevancy = AnswerRelevancyMetric(model=ollama_model)
context_precision = ContextualPrecisionMetric(model=ollama_model)

faithfulness.measure(test_case)
answer_relevancy.measure(test_case)
context_precision.measure(test_case)

print(f"Faithfulness: {faithfulness.score}")
print(f"Answer Relevancy: {answer_relevancy.score}")
print(f"Context Precision: {context_precision.score}")
```

## Summary Table

| Aspect | Recommendation |
|--------|----------------|
| **Provider** | Ollama for local/privacy; OpenAI for quality |
| **Embedding** | nomic-embed-text (free) or OpenAI text-embedding-3-small |
| **Chunk Size** | 500-1000 chars for general text; smaller for code |
| **Top-K** | 3-5 for simple Q&A; 5-10 for complex questions |
| **Retrieval** | Semantic search (default); hybrid for better recall |
| **Model** | llama3.2 for local; gpt-4o for quality |

---

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | Original RAG paper |
| [Dense Passage Retrieval (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.550/) | 2020 | DPR for retrieval (no arXiv version) |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [LangChain RetrievalQA](https://python.langchain.com/docs/modules/chains/) | QA chain documentation |
| [LangChain Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) | Chroma integration |
| [LlamaIndex VectorStoreIndex](https://docs.llamaindex.ai/en/stable/api/core/index/) | LlamaIndex index guide |
| [Ollama](https://github.com/ollama/ollama) | Local LLM execution |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) | Official LangChain tutorial |
| [Building RAG with LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/quickstart/) | LlamaIndex quickstart |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [LangChain](https://github.com/langchain-ai/langchain) | LangChain framework |
| [Ollama](https://github.com/ollama/ollama) | Local LLM runtime |

---

## Try It Yourself

Practice implementing Classic RAG with this notebook:

- [Classic RAG Implementation Notebook](../notebooks/01-classic-rag-implementation.ipynb)

---

*Next: [Knowledge Graph RAG](../2-architectures/kg-rag/)*
