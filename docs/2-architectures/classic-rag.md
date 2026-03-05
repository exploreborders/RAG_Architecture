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
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
    prompt=PROMPT,
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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama

# 1. Load Documents
documents = SimpleDirectoryReader("data").load_data()

# 2. Create Index
index = VectorStoreIndex.from_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

# 3. Configure Retriever
retriever = VectorRetriever(
    index=index,
    similarity_top_k=4
)

# 4. Create Query Engine (using Ollama)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=Ollama(model="llama3.2")
)

# 5. Query
response = query_engine.query("What is the main topic?")

print(response)
print("\nSources:")
for source in response.source_nodes:
    print(f"- {source.node.text[:100]}...")
```

## Unified Provider Setup (OpenAI + Ollama)

### Using the Provider Wrapper

```python
"""
Classic RAG with Unified Provider - Default is Ollama!
"""

# Import the provider wrapper
# See docs/3-technical/providers.md for full implementation

from enum import Enum

class Provider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

def create_rag_pipeline(provider: str = "ollama"):  # Default to Ollama!
    """Create RAG pipeline with any provider."""
    
    if provider == "ollama":
        # Ollama (local, free) - DEFAULT
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        
        llm = ChatOllama(model="llama3.2")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
    elif provider == "openai":
        # OpenAI (cloud)
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        
        llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain

# Usage - default is Ollama (free, local)!
rag_ollama = create_rag_pipeline()  # Uses Ollama by default
rag_openai = create_rag_pipeline("openai")  # Uses OpenAI if needed

# Same interface!
result1 = rag_ollama.run("What is RAG?")
result2 = rag_openai.run("What is RAG?")
```

## With Ollama (Local Models)

### Setup Ollama

```bash
# Install Ollama: https://ollama.ai

# Pull required models
ollama pull llama3.2      # LLM
ollama pull nomic-embed-text  # Embeddings

# Start server
ollama serve
```

### Complete Ollama Example

```python
"""
Classic RAG with Ollama (Local, Free)
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

# 3. Use Ollama embeddings (local, free)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # Free embedding model
    base_url="http://localhost:11434"
)

# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

# 5. Use Ollama LLM (local, free)
llm = ChatOllama(
    model="llama3.2",  # or "mistral", "codellama"
    base_url="http://localhost:11434",
    temperature=0
)

# 6. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 7. Query!
result = qa_chain.run("Your question here")
print(result)
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
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Evaluate RAG pipeline
results = evaluate(
    questions=questions,
    contexts=contexts,
    answers=answers,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(results)
```

---

*Next: [Knowledge Graph RAG](../2-architectures/kg-rag/)*
