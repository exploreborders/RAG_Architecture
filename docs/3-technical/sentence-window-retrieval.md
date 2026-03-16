# Sentence Window Retrieval

## Overview

Sentence window retrieval (also known as "small-to-big" or "parent-child" retrieval) is a technique where you retrieve smaller chunks (sentences) that match the query, then return the larger surrounding context (parent document) for generation.

## Why Sentence Window Retrieval

```
The Problem with Fixed Chunking:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original Document:
"The RAG architecture consists of three main components: the retriever,
the generator, and the database. The retriever searches for relevant
information. The generator creates the final response."

                 │
                 ▼
        Fixed Chunk (chunk_size=100):
        
        "The RAG architecture consists of three main components:
         the retriever, the generator, and the database. The retriever
         searches for relevant information."

                 │
                 ▼
        Problem: Context is incomplete!
        "The generator creates the final response" is missing

                 │
                 ▼
        Sentence Window Solution:
        
        Step 1: Retrieve matching SENTENCE
        → "The retriever searches for relevant information."
        
        Step 2: Return PARENT (larger context)
        → Full paragraph with complete context
```

## How It Works

```
Sentence Window Retrieval:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                    Sentence Window Retrieval                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query: "How does the retriever work?"                                  │
│                                                                         │
│  Step 1: Small Chunk Retrieval                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Document 1: "The retriever searches for relevant information."  │    │
│  │ Document 2: "The generator creates the final response."         │    │
│  │ ...                                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  Step 2: Expand to Parent Context                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Parent of Doc 1:                                                │    │
│  │ "The RAG architecture consists of three main components:        │    │
│  │ the retriever, the generator, and the database. The retriever   │    │
│  │ searches for relevant information. The generator creates        │    │
│  │ the final response."                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Step 3: Use expanded context for generation                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Basic Sentence Window Retriever

```python
"""
Sentence Window Retrieval Implementation
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

class SentenceWindowRetriever:
    """
    Sentence window retrieval with parent document expansion.
    
    Retrieves small chunks that match, then returns larger parent context.
    """
    
    def __init__(
        self,
        documents: list,
        embeddings,
        child_chunk_size: int = 200,
        child_overlap: int = 20,
        parent_chunk_size: int = 1000,
        parent_overlap: int = 100
    ):
        self.documents = documents
        self.embeddings = embeddings
        
        # Child splitter (smaller chunks - sentences)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        # Parent splitter (larger chunks)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n"]
        )
        
        # Build index
        self._build_index()
    
    def _build_index(self):
        """Build parent-child chunk indexes."""
        
        # Create parent chunks
        parent_docs = self.parent_splitter.split_documents(self.documents)
        
        # Create child chunks for each parent
        self.parent_to_children = {}
        self.child_to_parent = {}
        
        all_child_docs = []
        
        for parent in parent_docs:
            # Split parent into children
            children = self.child_splitter.split_documents([parent])
            
            # Map parent to children
            self.parent_to_children[parent.page_content] = children
            
            # Map each child to parent
            for child in children:
                self.child_to_parent[child.page_content] = parent
            
            all_child_docs.extend(children)
        
        # Create vector store with child chunks
        self.vectorstore = Chroma.from_documents(
            documents=all_child_docs,
            embedding=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def retrieve(self, query: str) -> list:
        """
        Retrieve using sentence window approach.
        
        1. Retrieve matching child (small) chunks
        2. Expand to parent (larger) context
        """
        
        # Step 1: Retrieve matching child chunks
        child_docs = self.retriever.invoke(query)
        
        # Step 2: Get unique parent documents
        parent_contents = set()
        parent_docs = []
        
        for child in child_docs:
            parent = self.child_to_parent.get(child.page_content)
            if parent and parent.page_content not in parent_contents:
                parent_contents.add(parent.page_content)
                parent_docs.append(parent)
        
        return parent_docs
```

### 2. Automatic Merging Retriever

A variant that automatically merges related chunks when they all retrieve similar content.

```python
"""
Automatic Merging Retrieval
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

class AutoMergingRetriever:
    """
    Automatically merge small chunks that are frequently retrieved together.
    
    This creates a hierarchy of chunks that can be merged on demand.
    """
    
    def __init__(
        self,
        documents: list,
        embeddings,
        chunk_sizes: list = None
    ):
        self.documents = documents
        self.embeddings = embeddings
        
        # Default: hierarchical chunk sizes
        self.chunk_sizes = chunk_sizes or [300, 600, 1200]
        
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Build hierarchical chunk structure."""
        
        self.level_stores = []
        
        for chunk_size in self.chunk_sizes:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 4
            )
            
            chunks = splitter.split_documents(self.documents)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            self.level_stores.append({
                "chunk_size": chunk_size,
                "vectorstore": vectorstore,
                "retriever": vectorstore.as_retriever()
            })
    
    def retrieve(self, query: str) -> list:
        """
        Retrieve with automatic merging.
        
        Start with smallest chunks, merge if multiple retrieve
        the same parent.
        """
        
        # Start with smallest chunks
        smallest_level = self.level_stores[0]
        initial_docs = smallest_level["retriever"].invoke(query)
        
        # Track which parent documents we've seen
        parent_map = {}
        
        for doc in initial_docs:
            # Check each level to find parent
            for level in self.level_stores[1:]:
                # This is simplified - real impl would track parent IDs
                parent_key = doc.page_content[:100]  # Simplified
                if parent_key not in parent_map:
                    parent_map[parent_key] = doc
        
        # Return merged results
        return list(parent_map.values())
```

### 3. LangChain Native Implementation

Using LangChain's built-in parent document retriever.

```python
"""
LangChain Parent Document Retriever
"""

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define child splitter (small chunks)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n", ". ", " "]
)

# Define parent splitter (large chunks)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n"]
)

# Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="sentence_window",
    embedding_function=embeddings
)

# Create parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=None,  # In-memory for demo
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 5}
)

# Add documents
retriever.add_documents(documents)
```

## Complete Example

```python
"""
Complete Sentence Window RAG Pipeline
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class SentenceWindowRAG:
    """RAG with sentence window retrieval."""
    
    def __init__(
        self,
        documents: list,
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text"
    ):
        # Initialize LLM and embeddings
        self.llm = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Create sentence window retriever
        self.retriever = SentenceWindowRetriever(
            documents=documents,
            embeddings=self.embeddings,
            child_chunk_size=200,
            parent_chunk_size=1000
        )
        
        # Define prompt
        self.prompt = PromptTemplate.from_template("""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:""")
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        
        # Retrieve with sentence window
        relevant_docs = self.retriever.retrieve(question)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer
        response = self.llm.invoke(
            self.prompt.format(context=context, question=question)
        )
        
        return {
            "answer": response.content,
            "sources": [doc.page_content[:200] for doc in relevant_docs]
        }
```

## When to Use Sentence Window Retrieval

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Technical docs with code** | Yes | Preserve complete code blocks |
| **Legal documents** | Yes | Maintain complete clauses |
| **Long articles** | Yes | Preserve narrative flow |
| **Q&A with short answers** | No | Overkill, simple chunking fine |
| **Conversational data** | Maybe | Depends on turn structure |

## Comparison

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fixed chunks** | Simple, fast | May break context | Simple Q&A |
| **Sentence window** | Preserves context | More complex | Technical docs |
| **Auto-merging** | Adaptive | Complex | Variable content |
| **Hierarchical** | Flexible | Slow | Large corpora |

## References

### Implementations

| Resource | Description |
|----------|-------------|
| [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever) | Official LangChain implementation |
| [LlamaIndex Sentence Window](https://docs.llamaindex.ai/en/latest/examples/node_postprocessor/sentence_windowing/) | LlamaIndex implementation |

### Tutorials & Blogs

| Blog | Description |
|------|-------------|
| [Weaviate Advanced RAG](https://weaviate.io/blog/advanced-rag) | Includes sentence window techniques |
| [Neo4j Advanced RAG](https://neo4j.com/blog/genai/advanced-rag-techniques/) | Chunking strategies comparison |

---

*Next: [Embeddings Strategies](embeddings-strategies.md)*
