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
                        │  Embedding Model     │
                        │  (Convert query to   │
                        │   vector)            │
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

### 3. Embedding Model
```python
from langchain_community.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
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

### 6. QA Chain
```python
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

## Complete Example (LangChain)

```python
"""
Classic RAG Implementation with LangChain
"""

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
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

# 3. Create Embeddings
embeddings = OpenAIEmbeddings()

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

# 7. Create QA Chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
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

## Complete Example (LlamaIndex)

```python
"""
Classic RAG Implementation with LlamaIndex
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

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

# 4. Create Query Engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=OpenAI(model="gpt-4")
)

# 5. Query
response = query_engine.query("What is the main topic?")

print(response)
print("\nSources:")
for source in response.source_nodes:
    print(f"- {source.node.text[:100]}...")
```

## With Ollama (Local Models)

```python
"""
Classic RAG with Ollama (Local LLM + Embeddings)
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Use Ollama embeddings (local)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

# Use Ollama LLM (local)
llm = ChatOllama(model="llama3.2")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

result = qa_chain.run("Your question here")
print(result)
```

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
