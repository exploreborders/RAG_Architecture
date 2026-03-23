# Knowledge Graph RAG (KG-RAG)

## Overview

**Knowledge Graph RAG (KG-RAG)** enhances traditional RAG by incorporating structured knowledge graphs for retrieval. This enables reasoning over relationships and complex queries that require understanding of entity connections.

## Why Knowledge Graphs?

### Limitations of Vector-Only RAG

| Aspect | Vector RAG | KG-RAG |
|--------|------------|--------|
| **Relationships** | Implicit, embedding-based | Explicit, queryable |
| **Reasoning** | Limited | Graph traversal |
| **Explainability** | Black box | Traceable paths |
| **Updates** | Re-index entire corpus | Add triples |

### When KG-RAG Shines

```
Example Query Requiring Graph Reasoning:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Who are the competitors of Company X that are 
located in the same country as their suppliers?"

Vector RAG: May retrieve relevant documents but struggle to 
            connect the relationships explicitly

KG-RAG: Traverse: Company X → suppliers → location ← competitors
        Returns structured, connected results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Architecture

```
KG-RAG Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        ┌─────────────────────┐
                        │   User Query        │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Query Decomposition│
                        │   (Entity/Relation   │
                        │    Extraction)       │
                        └──────────┬───────────┘
                                   │
                                   ▼
              ┌────────────────────┴─────────────────┐
              │                                      │
              ▼                                      ▼
    ┌─────────────────┐                     ┌─────────────────┐
    │  Vector Search  │                     │  Graph Traversal│
    │  (Semantic)     │                     │  (Cypher/SPARQL)│
    └────────┬────────┘                     └────────┬────────┘
             │                                       │
             └─────────────────┬─────────────────────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │   Result Fusion     │
                     │   & Ranking         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   LLM Generation    │
                     └─────────────────────┘
```

## Implementation with LangChain + Knowledge Graph

### Setup Knowledge Graph

```python
"""
KG-RAG Implementation using LangChain + NetworkX + Neo4j
"""

from langchain_community.graphs import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from pydantic import BaseModel
from typing import List, Tuple

# Initialize Neo4j Knowledge Graph
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)
```

### Extract Knowledge Graph from Documents

```python
"""
Knowledge Graph Extraction Pipeline
"""

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore

# Extract entities and relationships using LLM
llm = ChatOllama(model="llama3.2")

graph_transformer = LLMGraphTransformer(llm=llm)

# Process documents to extract graph
def extract_graph_from_documents(documents: List) -> Neo4jGraph:
    """Extract knowledge graph from documents."""
    
    # 1. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    # 2. Extract graph using LLM
    graph_documents = graph_transformer.convert_to_graph_documents(chunks)
    
    # 3. Store in Neo4j
    graph.add_graph_documents(graph_documents)
    
    return graph

# Example graph schema
graph_query = """
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE
"""

# Define relationships
relationships = """
MATCH (a:Entity), (b:Entity)
WHERE a.name CONTAINS 'Company' AND b.name CONTAINS 'Product'
CREATE (a)-[:PRODUCES]->(b)
"""
```

### Hybrid Vector + Graph Retrieval

```python
"""
Hybrid Retrieval: Vector Search + Graph Traversal
"""

from langchain_core.documents import Document

class HybridKGRetriever:
    """Combines vector and graph retrieval."""
    
    def __init__(self, vectorstore, graph, llm, top_k=5):
        self.vectorstore = vectorstore
        self.graph = graph
        self.llm = llm
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Document]:
        """Execute hybrid retrieval."""
        
        # 1. Extract entities from query
        entities = self._extract_entities(query)
        
        # 2. Graph retrieval
        graph_results = self._graph_retrieve(entities)
        
        # 3. Vector retrieval
        vector_results = self.vectorstore.similarity_search(
            query, k=self.top_k
        )
        
        # 4. Fuse and rank results
        combined = self._combine_results(
            graph_results, 
            vector_results,
            query
        )
        
        return combined[:self.top_k]
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query."""
        
        prompt = f"""
        Extract key entities (people, organizations, products, 
        locations) from this query: {query}
        
        Return as comma-separated list.
        """
        
        result = self.llm.invoke(prompt)
        return [e.strip() for e in result.split(",")]
    
    def _graph_retrieve(self, entities: List[str]) -> List[Document]:
        """Retrieve from knowledge graph."""
        
        results = []
        
        for entity in entities:
            # Cypher query for graph traversal
            cypher = f"""
            MATCH path = (e:Entity)-[r*1..2]-(related)
            WHERE e.name CONTAINS '{entity}'
            RETURN path, e, r, related
            LIMIT 5
            """
            
            graph_data = self.graph.query(cypher)
            
            # Convert to documents
            for item in graph_data:
                doc = Document(
                    page_content=f"Graph path: {item}",
                    metadata={"source": "knowledge_graph", "entity": entity}
                )
                results.append(doc)
        
        return results
    
    def _combine_results(self, graph_results, vector_results, query):
        """Combine and re-rank results."""
        
        # Simple fusion: concatenate with priority
        combined = []
        
        # Add graph results first (higher weight for relationships)
        combined.extend(graph_results)
        
        # Add vector results
        combined.extend(vector_results)
        
        return combined

# Usage
hybrid_retriever = HybridKGRetriever(
    vectorstore=vectorstore,
    graph=graph,
    llm=llm
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,
    return_source_documents=True
)
```

### Complete KG-RAG Example

```python
"""
Complete KG-RAG Pipeline
"""

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

class KGRAGPipeline:
    """End-to-end Knowledge Graph RAG pipeline."""
    
    def __init__(self, neo4j_config: dict):
        # Initialize components
        self.graph = Neo4jGraph(**neo4j_config)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.2")
        self.vectorstore = None
        self.qa_chain = None
    
    def build_index(self, documents, chunk_size=1000):
        """Build knowledge graph and vector index."""
        
        # 1. Create vector store
        self.vectorstore = Neo4jVector.from_documents(
            documents=documents,
            embedding=self.embeddings,
            graph=self.graph
        )
        
        # 2. Extract knowledge graph
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size
        )
        chunks = splitter.split_documents(documents)
        
        graph_transformer = LLMGraphTransformer(llm=self.llm)
        graph_docs = graph_transformer.convert_to_graph_documents(chunks)
        
        self.graph.add_graph_documents(graph_docs)
        
        # 3. Create QA chain
        retriever = self.vectorstore.as_retriever()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question: str) -> dict:
        """Query the KG-RAG system."""
        
        return self.qa_chain({"query": question})

# Usage
config = {
    "neo4j_config": {
        "url": "bolt://localhost:7687",
        "username": "neo4j", 
        "password": "your_password"
    },
    "openai_config": {
        "openai_api_key": "your_key"
    }
}

pipeline = KGRAGPipeline(**config)
pipeline.build_index(documents)
result = pipeline.query("What products does Company X produce?")
```

## Implementation with LlamaIndex

```python
"""
KG-RAG with LlamaIndex and Neo4j
"""

from llama_index.core import KnowledgeGraphIndex
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create knowledge graph index
llm = OpenAI(model="gpt-4")

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets=10,
    llm=llm
)

# Create KG retriever
kg_retriever = KnowledgeGraphRAGRetriever(
    index=kg_index,
    llm=llm,
    verbose=True
)

# Query
query_engine = kg_index.as_query_engine(
    retriever=kg_retriever
)

response = query_engine.query("Your question about relationships?")
```

## KG-RAG with LangGraph

```python
"""
Agentic KG-RAG using LangGraph
"""

from langgraph.graph import StateGraph
from langchain_community.graphs import Neo4jGraph
from pydantic import BaseModel
from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    entities: List[str]
    graph_results: List
    vector_results: List
    final_answer: str

# Initialize graph
graph = StateGraph(GraphState)

# Node 1: Extract Entities
def extract_entities(state: GraphState):
    """Extract entities using LLM."""
    # Entity extraction logic
    entities = llm.invoke(
        f"Extract entities from: {state['question']}"
    )
    return {"entities": entities.split(",")}

# Node 2: Graph Retrieval
def graph_retrieve(state: GraphState):
    """Retrieve from knowledge graph."""
    results = []
    for entity in state["entities"]:
        cypher = f"MATCH (e)-[r]-(related) WHERE e.name='{entity}' RETURN e,r,related"
        results.extend(graph.query(cypher))
    return {"graph_results": results}

# Node 3: Vector Retrieval
def vector_retrieve(state: GraphState):
    """Retrieve from vector store."""
    results = vectorstore.similarity_search(state["question"], k=3)
    return {"vector_results": results}

# Node 4: Generate Answer
def generate(state: GraphState):
    """Generate final answer."""
    context = state["graph_results"] + state["vector_results"]
    prompt = f"Answer based on: {context}\n\nQuestion: {state['question']}"
    answer = llm.invoke(prompt)
    return {"final_answer": answer}

# Build workflow
graph.add_node("extract", extract_entities)
graph.add_node("kg_retrieve", graph_retrieve)
graph.add_node("vec_retrieve", vector_retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("extract")
graph.add_edge("extract", "kg_retrieve")
graph.add_edge("kg_retrieve", "vec_retrieve")
graph.add_edge("vec_retrieve", "generate")
graph.set_finish_point("generate")

app = graph.compile()
```

## Pros and Cons

### ✅ Advantages

| Advantage | Description |
|-----------|-------------|
| **Relationship Understanding** | Explicit graph relationships |
| **Explainability** | Traceable reasoning paths |
| **Multi-hop Queries** | Graph traversal enables multi-step |
| **Structured Reasoning** | Cypher/SPARQL queries |
| **Better for Domains** | Complex relationships (medical, legal) |

### ❌ Limitations

| Limitation | Description |
|------------|-------------|
| **Complex Setup** | Requires KG database (Neo4j, etc.) |
| **KG Maintenance** | Graph must be kept updated |
| **Entity Extraction** | Requires good NER/LLM extraction |
| **Higher Cost** | More components to maintain |
| **Limited Semantic** | Graph queries may miss semantic matches |

## When to Use KG-RAG

### ✅ Best For

- Domain-specific queries with relationships
- Organizational charts, hierarchies
- Complex multi-hop questions
- When explainability is critical
- Medical, legal, financial applications
- Product catalogs with relationships

### ❌ Not Ideal For

- Simple Q&A without relationships
- When KG maintenance overhead is too high
- Rapid prototyping (use Classic RAG first)
- Limited structured data scenarios

## Evaluation Metrics

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, FaithfulnessMetric
from deepeval.models import OllamaModel

# Initialize with Ollama
ollama_model = OllamaModel(model="llama3.2")

# Create test case
test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG stands for...",
    retrieval_context=["RAG is...", "It combines..."]
)

# Knowledge graph specific metrics
context_recall = ContextualRecallMetric(model=ollama_model)
context_precision = ContextualPrecisionMetric(model=ollama_model)
faithfulness = FaithfulnessMetric(model=ollama_model)

context_recall.measure(test_case)
context_precision.measure(test_case)
faithfulness.measure(test_case)

print(f"Context Recall: {context_recall.score}")
print(f"Context Precision: {context_precision.score}")
print(f"Faithfulness: {faithfulness.score}")
```

---

## Try It Yourself

Practice implementing Knowledge Graph RAG with this notebook:

- [KG-RAG Implementation Notebook](../notebooks/02-kg-rag-implementation.ipynb)

---

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [From Local to Global: GraphRAG Approach](https://www.microsoft.com/en-us/research/publication/graphrag/) | 2024 | Microsoft's GraphRAG (Research Blog) |
| [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921) | 2024 | Comprehensive GraphRAG survey (327 citations) |
| [GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](https://arxiv.org/abs/2502.01113) | 2025 | Graph foundation models for RAG |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [Neo4j Graph Database](https://neo4j.com/docs/) | Neo4j documentation |
| [LangChain Neo4j](https://python.langchain.com/docs/integrationsgraphs/neo4j_graph) | LangChain Neo4j integration |
| [LlamaIndex Knowledge Graph](https://docs.llamaindex.ai/en/stable/module_ides/knowledge_graph/) | LlamaIndex KG guide |
| [GraphRAG Microsoft](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-knowledge/) | Microsoft GraphRAG blog |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [Neo4j Advanced RAG](https://neo4j.com/blog/genai/advanced-rag-techniques/) | 15 advanced techniques |
| [Building Knowledge Graphs](https://towardsdatascience.com/building-knowledge-graphs-a-technical-deep-dive-ddc2dc76e79c) | KG construction guide |
| [GraphRAG vs Vector RAG](https://www.pinecone.io/learn/graph-rag-vs-vector-rag) | Comparison guide |

### GitHub Repositories

| Repo | Description |
|------|-------------|
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | Microsoft's GraphRAG |
| [neo4j-graphrag](https://github.com/neo4j/graphrag) | Neo4j GraphRAG package |
| [LangChain Graph Transformers](https://github.com/langchain-ai/langchain/tree/master/libs/langchain-experimental) | LLM graph extraction |

---

*Next: [Agentic RAG](../2-architectures/agentic-rag/)*
