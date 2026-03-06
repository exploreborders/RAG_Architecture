# Chunking Strategies

## Overview

**Chunking** is the process of splitting documents into smaller, manageable pieces for embedding and retrieval. Choosing the right chunking strategy significantly impacts RAG performance.

## Why Chunking Matters

```
Impact of Chunking:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Too Small                          Too Large
┌─────────────┐                    ┌──────────────┐
│ • Loss of   │                    │ • Too much   │
│   context   │  OPTIMAL           │   noise      │
│ • Fragmented│◄──────────────────►│ • Lost focus │
│   meaning   │    CHUNK           │ • Higher     │
│ • Poor      │    SIZE            │   cost       │
│   retrieval │                    │ • Cuts off   │
└─────────────┘                    │   concepts   │
                                   └──────────────┘
```

## Chunking Strategies

### 1. Fixed-Size Chunking

```python
"""
Fixed-Size Chunking
"""

from langchain_text_splitters import CharacterTextSplitter

# Simple character-based
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,  # Overlap between chunks
    separator="\n"      # Split on newlines
)

chunks = splitter.split_text(large_text)
```

**Pros**: Simple, predictable
**Cons**: May break semantic units

### 2. Recursive Chunking

```python
"""
Recursive Chunking - Try multiple separators
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # Try separators in order
    separators=[
        "\n\n",      # Paragraphs first
        "\n",        # Then lines
        ". ",        # Then sentences
        " ",         # Then words
        ""           # Then characters
    ]
)

chunks = splitter.split_documents(documents)
```

**Pros**: Preserves semantic boundaries
**Cons**: May still break code/tables

### 3. Semantic Chunking

```python
"""
Semantic Chunking - Split by meaning
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

# Uses embeddings to find semantic breaks
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=0.95
)

chunks = semantic_splitter.split_text(text)
```

**Pros**: Respects semantic boundaries
**Cons**: Slower, requires embedding model

### 4. Sentence-Based Chunking

```python
"""
Sentence-Level Chunking
"""

from langchain_text_splitters import NLTKTextSplitter
# or
from langchain_text_splitters import SpacyTextSplitter

# Using NLTK
nltk_splitter = NLTKTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = nltk_splitter.split_text(text)

# Using spaCy (better for many languages)
spacy_splitter = SpacyTextSplitter(
    pipeline="en_core_web_sm",
    chunk_size=1000,
    chunk_overlap=200
)

chunks = spacy_splitter.split_text(text)
```

**Pros**: Preserves complete sentences
**Cons**: May vary in size significantly

### 5. Markdown/HTML Chunking

```python
"""
Structure-Aware Chunking for Markdown
"""

from langchain_text_splitters import MarkdownTextSplitter

markdown_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = markdown_splitter.split_text(markdown_content)

# For HTML
from langchain_text_splitters import HTMLHeaderTextSplitter

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "header1"),
        ("h2", "header2"),
        ("h3", "header3"),
    ]
)

chunks = html_splitter.split_text(html_content)
```

**Pros**: Preserves document structure
**Cons**: Loses cross-section context

### 6. Code-Aware Chunking

```python
"""
Programming Language Chunking
"""

from langchain_text_splitters import Language

# Support for many languages
code_splitters = {}

for lang in [
    Language.PYTHON,
    Language.JAVASCRIPT,
    Language.TYPESCRIPT,
    Language.HTML,
    Language.CSS,
    Language.JAVA,
    Language.CPP,
]:
    code_splitters[lang] = RecursiveCharacterTextSplitter.from_language(
        language=lang,
        chunk_size=1000,
        chunk_overlap=200
    )

# Python example
python_chunks = code_splitters[Language.PYTHON].split_text(python_code)
```

### 7. Agentic Chunking (Advanced)

```python
"""
LLM-Guided Intelligent Chunking
"""

from langchain_ollama import ChatOllama

class IntelligentChunking:
    """Use LLM to determine optimal chunks."""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3.2")
    
    def split(self, text: str, max_chunk_size: int = 1000) -> list:
        """Intelligently split text."""
        
        prompt = f"""Analyze this text and suggest logical chunk boundaries.
        
Text:
{text[:2000]}... (truncated for prompt)

Respond with JSON array of chunks, each containing:
- "content": the text for that chunk
- "reason": why this is a logical breaking point

Max chunk size: {max_chunk_size} characters"""

        response = self.llm.invoke(prompt)
        chunks = json.loads(response)
        
        return [c["content"] for c in chunks]
```

## Choosing Chunk Size

### Factors to Consider

| Factor              | Smaller Chunks             | Larger Chunks            |
|---------------------|----------------------------|--------------------------|
| **Query Type**      | Specific facts             | Broad topics             | 
| **Context Needed**  | Precise answers            | Full understanding       |
| **Embedding Model** | Models have context limits | Better with more context |
| **Compute Cost**    | Lower                      | Higher                   |

### Experimentation Guide

```python
"""
Finding Optimal Chunk Size
"""

def evaluate_chunk_size(rag_pipeline, test_queries: list, chunk_sizes: list) -> dict:
    """Evaluate different chunk sizes."""
    
    results = {}
    
    for size in chunk_sizes:
        # Rebuild index with this chunk size
        splitter = RecursiveCharacterTextSplitter(chunk_size=size)
        chunks = splitter.split_documents(documents)
        
        # Build vector store
        vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # Evaluate
        eval_results = evaluate_chunking(
            rag_pipeline, 
            test_queries,
            vectorstore
        )
        
        results[size] = {
            "avg_score": sum(eval_results) / len(eval_results),
            "chunks_created": len(chunks),
            "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks)
        }
    
    return results

# Common sizes to test
chunk_sizes = [256, 512, 768, 1024, 1536, 2048]

results = evaluate_chunk_size(qa_chain, test_queries, chunk_sizes)

# Analyze results
for size, metrics in results.items():
    print(f"Size {size}: Score={metrics['avg_score']:.3f}, Chunks={metrics['chunks_created']}")
```

## Best Practices

### 1. Consider Your Data

```python
# Different strategies for different data

def get_chunking_strategy(document_type: str):
    """Select chunking based on document type."""
    
    strategies = {
        "markdown": MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200),
        "code": RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON, chunk_size=1000, chunk_overlap=200
        ),
        "plain_text": RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        ),
        "html": HTMLHeaderTextSplitter(),
        "pdf": # Use specialized PDF loader
    }
    
    return strategies.get(document_type, strategies["plain_text"])
```

### 2. Add Metadata

```python
"""
Enrich chunks with metadata
"""

from langchain_core.documents import Document

def create_enriched_chunks(documents: list) -> list:
    """Add useful metadata to chunks."""
    
    enriched = []
    
    for doc in documents:
        # Split
        chunks = splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            # Add metadata
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_title": doc.metadata.get("title", "Unknown"),
                "doc_source": doc.metadata.get("source", "Unknown"),
                "char_count": len(chunk.page_content),
            })
            enriched.append(chunk)
    
    return enriched
```

### 3. Hybrid Approaches

```python
"""
Combine multiple chunking strategies
"""

class HybridChunker:
    """Combine multiple chunking approaches."""
    
    def __init__(self):
        # Primary: semantic
        self.semantic = SemanticChunker(
            embeddings=OpenAIEmbeddings(),
            breakpoint_threshold_amount=0.8
        )
        # Fallback: recursive
        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def split(self, text: str) -> list:
        # Try semantic first
        semantic_chunks = self.semantic.split_text(text)
        
        # Check if any chunks are too large
        result = []
        for chunk in semantic_chunks:
            if len(chunk) > 1500:
                # Split oversized chunks recursively
                result.extend(self.recursive.split_text(chunk))
            else:
                result.append(chunk)
        
        return result
```

## Summary Table

| Strategy      | Best For         | Trade-offs          |
|---------------|------------------|---------------------|
| **Fixed**     | Simple documents | May break semantics |
| **Recursive** | General purpose  | Good default        |
| **Semantic**  | Quality-critical | Slower              |
| **Sentence**  | Natural text     | Variable size       |
| **Markdown**  | Documentation    | Loses cross-section |
| **Code**      | Repositories     | Language-specific   |

---

*Next: [Query Optimization](./query-optimization.md)*
