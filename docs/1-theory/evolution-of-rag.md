# Evolution of RAG

## Timeline Overview

```
RAG Evolution (2020 - Present)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2020          2021-2022           2023            2024-2025
  │               │                │                 │
  ▼               ▼                ▼                 ▼
┌────┐         ┌──────┐         ┌───────┐         ┌───────┐
│Meta│────────►│OpenAI│────────►│Modular│────────►│Agentic│
│RAG │         │DSP   │         │RAG    │         │RAG    │
│    │         │      │         │       │         │       │
└────┘         └──────┘         └───────┘         └───────┘
   │              │                 │                 │
   ▼              ▼                 ▼                 ▼
Naive          Advanced          Hybrid &          Self-RAG
RAG            Retrieval         Context           Agentic
                                 Enhanc.           RAG
```

## 2020: The Birth of RAG

### Meta's Landmark Paper

**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)

Key innovations:
- First formal RAG architecture
- Combined dense retrieval (DPR - Dense Passage Retrieval) with seq2seq (sequence-to-sequence) generation
- Pre-trained on large-scale knowledge bases

> **Paper**: [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

### Architecture

```
Original RAG (2020):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query ──► Dense Retriever ──► Top-K Docs ──► Generator
              (DPR)             │            (BART/T5)
                                │
                         ┌──────┴──────┐
                         │  Wikipedia  │
                         │  Index      │
                         └─────────────┘
```

### Technical Details

| Component | Original RAG | Description |
|-----------|--------------|-------------|
| Retriever | DPR | Dense Passage Retriever using BERT |
| Index | Wikipedia | 21M passages |
| Generator | BART-large | 400M parameters |
| Top-K | 10 | Number of retrieved passages |

### Impact

- Established baseline for all future RAG research
- Showed significant improvements over pure generation
- Introduced the "retrieve-then-generate" paradigm

> **Summary (2020)**: The foundational RAG paper established the core pattern of retrieving relevant documents and using them as context for generation. This basic architecture became the baseline for all future improvements.

### Key Papers from 2020

| Paper | Contribution |
|-------|-------------|
| [RAG](https://arxiv.org/abs/2005.11401) | Original RAG architecture |
| [DPR](https://arxiv.org/abs/2004.04906) | Dense Passage Retrieval |
| [BART](https://arxiv.org/abs/1910.13461) | Seq2seq denoising autoencoder |

## 2021-2022: Advanced Retrieval

*Building on the foundational RAG architecture, researchers focused on improving the retrieval component itself.*

### Key Developments

| Year | Innovation | Description |
|------|------------|-------------|
| 2020 | DPR Improvements | Better dense retrievers |
| 2021 | BM25 + Dense | Hybrid search emergence (keyword + embedding) |
| 2022 | In-context RAG | Few-shot learning with retrieval |
| 2022 | Atlas | Fine-tuned retriever + generator |

### The Rise of Hybrid Search

Combining keyword and semantic search:
- **BM25**: Traditional keyword-based ranking, robust for exact matches
- **Dense (Embedding)**: Semantic understanding using neural networks
- **Hybrid**: Best of both worlds - combines keyword precision with semantic understanding

### Key Papers 2021-2022

| Paper | Year | Contribution |
|-------|------|--------------|
| [ColBERT](https://arxiv.org/abs/2007.00814) | 2020 | Late interaction retrieval |
| [ANCE](https://arxiv.org/abs/2007.00808) | 2020 | Approximate nearest neighbor training |
| [RocketQA](https://arxiv.org/abs/2010.08191) | 2020 | Cross-encoder distillation |
| [Atlas](https://arxiv.org/abs/2208.03565) | 2022 | Fine-tuned retriever+generator |

### Technical Innovations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Retrieval Innovations 2021-2022                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Late Interaction (ColBERT)                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  Query embeddings interact with document tokens post-embedding              │
│  → Better than single-vector similarity                                     │
│                                                                             │
│  2. Hard Negative Mining (ANCE)                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  Train retrievers using difficult negative examples                         │
│  → More robust retrieval                                                    │
│                                                                             │
│  3. Cross-Encoder Reranking                                                 │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  Two-stage: retrieve broadly → rerank precisely                             │
│  → Higher precision                                                         │
│                                                                             │
│  4. Hybrid Search                                                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  BM25 + Dense = keyword precision + semantic depth                         │
│  → Best of both worlds                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code Example: Hybrid Search (2022 Pattern)

```python
"""
Hybrid Search Implementation (2021-2022)
Combines BM25 (keyword) with Dense (semantic) retrieval
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

class HybridSearch:
    """Hybrid search combining keyword and semantic."""
    
    def __init__(self, documents, embeddings):
        # Dense retriever (semantic)
        self.dense_store = Chroma.from_documents(
            documents, embeddings
        )
        self.dense_retriever = self.dense_store.as_retriever()
        
        # Keyword retriever (BM25)
        self.keyword_retriever = BM25Retriever.from_documents(
            documents
        )
        
        # Ensemble with weights
        self.ensemble = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.keyword_retriever],
            weights=[0.7, 0.3]  # 70% semantic, 30% keyword
        )
    
    def retrieve(self, query, k=5):
        """Hybrid retrieval."""
        return self.ensemble.invoke(query)
```

> **Summary (2021-2022)**: The focus shifted from basic retrieval to combining multiple search methods. Hybrid search emerged as a best practice, and researchers began exploring fine-tuned models specifically for retrieval tasks.

## 2023: Modular RAG Revolution

*2023 marked the shift from linear pipelines to modular, composable architectures.*

### Key Innovations

| Technique | Description | Paper |
|-----------|-------------|-------|
| **Query Rewriting** | Improve query before retrieval | [2023] |
| **Reranking** | Two-stage scoring | [ColBERTv2](https://arxiv.org/abs/2112.01488) |
| **Chunking Strategies** | Optimize context windows | [2023] |
| **Self-Ask** | LLM asks follow-up questions | [Self-Ask](https://arxiv.org/abs/2210.05050) |
| **Active RAG** | Decide when to retrieve | [Active RAG](https://arxiv.org/abs/2305.10794) |

### Modular Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Modular RAG Components (2023)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                        │
│  │   Input     │──►│  Retrieval  │──►│   Router    │                        │
│  │ Processing  │   │   Module    │   │   Module    │                        │
│  └─────────────┘   └─────────────┘   └─────────────┘                        │
│                                            │                                │
│                    ┌─────────────┐         │                                │
│                    │   Fusion    │◄────────┤                                │
│                    │   Module    │         │                                │
│                    └─────────────┘   ┌─────────────┐                        │
│                           │          │   Memory    │                        │
│                           ▼          │   Module    │                        │
│                    ┌─────────────┐   └─────────────┘                        │
│                    │  Generator  │                                          │
│                    │   Module    │                                          │
│                    └─────────────┘                                          │
│                                                                             │
│  New Components in 2023:                                                    │
│  • Query rewriting transformers                                             │
│  • Cross-encoder rerankers                                                  │
│  • Parent document retrievers                                               │
│  • Hypothetical document embeddings (HyDE)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Papers 2023

| Paper | Year | Contribution |
|-------|------|--------------|
| [HyDE](https://arxiv.org/abs/2212.10496) | 2022/23 | Hypothetical document embeddings |
| [Self-Ask](https://arxiv.org/abs/2210.05050) | 2022/23 | LLM asks follow-up questions |
| [Active RAG](https://arxiv.org/abs/2305.10794) | 2023 | Active retrieval decision |
| [Chain-of-Note](https://arxiv.org/abs/2311.09210) | 2023 | Reading notes for retrieval |

## 2024-2025: Agentic & Advanced RAG

*The next leap came with autonomous systems that could decide when and how to retrieve information.*

### Self-RAG (2024)

**"Self-RAG: Learning to Retrieve, Generate, and Critique"** (Asai et al., 2024)

- Model learns to retrieve when needed
- Generates reflection tokens
- Self-critiques its outputs

> **Paper**: [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
> **GitHub**: [https://github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)

```
Self-RAG Process:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query ──► [Is retrieval needed?] ──► Yes ──► Retrieve ──► Generate ──► Critique
              │
              │
              └─ No ──► Direct Generation ──► Critique
```

### Agentic RAG (2024-2025)

**AI agents with autonomous control** over the RAG pipeline:

- **Planning**: Break down complex queries
- **Tool Use**: Select appropriate retrieval strategies
- **Reflection**: Evaluate and refine outputs
- **Memory**: Maintain conversation context

### Graph RAG (Microsoft)

**"From Local to Global: A Graph RAG Approach"** (Microsoft Research, 2024)

- Builds knowledge graphs from documents
- Community summarization
- Global knowledge synthesis

> **Blog**: [GraphRAG: Unlocking LLM knowledge](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
> **GitHub**: [microsoft/graphrag](https://github.com/microsoft/graphrag)

### Multimodal RAG (2025)

**Extending RAG beyond text**:
- Images, Video, Audio
- Cross-modal retrieval
- Unified embedding spaces

### Key Papers 2024-2025

| Paper | Year | Contribution |
|-------|------|--------------|
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 2024 | Self-reflective retrieval |
| [Corrective RAG](https://arxiv.org/abs/2401.15884) | 2024 | Error detection/correction |
| [Agentic RAG Survey](https://arxiv.org/abs/2501.09136) | 2025 | Agentic systems overview |
| [Comprehensive RAG Survey](https://arxiv.org/abs/2506.00054) | 2025 | All RAG architectures |
| [GFM-RAG](https://arxiv.org/abs/2502.01113) | 2025 | Graph foundation models |

### Code Example: Agentic RAG (2024 Pattern)

```python
"""
Agentic RAG Implementation (2024-2025)
Uses agents for autonomous retrieval decisions
"""

from langgraph.graph import StateGraph
from langchain_core.tools import tool

class AgenticRAG:
    """Agent-based RAG with autonomous decisions."""
    
    def __init__(self, llm, vectorstore, tools):
        self.llm = llm
        self.vectorstore = vectorstore
        self.tools = tools
        
        # Build agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the agent workflow graph."""
        
        graph = StateGraph(AgentState)
        
        # Define nodes
        graph.add_node("analyze", self.analyze_query)
        graph.add_node("retrieve", self.retrieve_docs)
        graph.add_node("generate", self.generate_response)
        graph.add_node("evaluate", self.evaluate_response)
        
        # Define edges
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_conditional_edges(
            "evaluate",
            self.should_redo,
            {
                "redo": "retrieve",
                "done": END
            }
        )
        
        return graph.compile()
    
    def analyze_query(self, state):
        """Analyze query complexity and plan approach."""
        query = state["query"]
        
        # LLM decides: simple or complex?
        prompt = f"""Is this query simple or complex?
        Complex = needs multiple steps, multiple sources, or reasoning.
        
        Query: {query}
        
        Respond with ONLY: simple or complex"""
        
        result = self.llm.invoke(prompt)
        state["approach"] = result.content
        
        return state
    
    def retrieve_docs(self, state):
        """Retrieve based on approach."""
        query = state["query"]
        
        if state["approach"] == "simple":
            docs = self.vectorstore.similarity_search(query, k=3)
        else:
            # Multi-step retrieval for complex queries
            docs = self._multi_step_retrieve(query)
        
        state["documents"] = docs
        return state
    
    def generate_response(self, state):
        """Generate with retrieved context."""
        # ... generation logic
        return state
    
    def evaluate_response(self, state):
        """Evaluate if response is satisfactory."""
        # ... evaluation logic
        state["satisfactory"] = True  # or False
        return state
    
    def should_redo(self, state):
        """Decide whether to redo retrieval."""
        return "done" if state["satisfactory"] else "redo"
```

> **Summary (2024-2025)**: RAG became autonomous with Self-RAG and agentic systems. Graph RAG emerged for global knowledge understanding, and multimodal capabilities extended RAG beyond text.

## Current State (2026)

### Architecture Spectrum

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Architecture Spectrum                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Naive RAG ──► Advanced RAG ──► Modular RAG ──► Agentic RAG │
│      │              │               │              │        │
│   Basic         Enhanced        Composable    Autonomous    │
│   Pipeline      Retrieval       Components    Agents        │
│                                                             │
│   Simple Q&A    Complex         Flexible      Multi-step    │
│                 Queries         Workflows     Reasoning     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Future Directions

### Emerging Patterns

1. **Anticipatory RAG**
   - Predicts knowledge needs before query completion
   
2. **Corrective RAG**
   - Detects and corrects retrieval/generation errors
   
3. **Speculative RAG**
   - Parallel retrieval of multiple strategies
   
4. **Continuous Learning**
   - Online updating of knowledge bases

### Detailed Future Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Emerging RAG Patterns                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Anticipatory RAG                                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│  Query: "Tell me about mach..."                                             │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ Predict: "machine│                                                       │
│  │ learning" needs  │                                                       │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  Pre-retrieve relevant documents BEFORE user finishes typing!               │
│                                                                             │
│  ─────────────────────────────────────────────────────────────              │
│                                                                             │
│  2. Corrective RAG                                                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Generate    │───►│   Evaluate   │───►│  Correct     │                   │
│  │  Response    │    │  (check      │    │  if needed   │                   │
│  └──────────────┘    │  grounded)   │    └──────────────┘                   │
│                      └──────────────┘                                       │
│                                                                             │
│  ─────────────────────────────────────────────────────────────              │
│                                                                             │
│  3. Speculative RAG                                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│                                                                             │
│  Query ──► Parallel retrievals (semantic, keyword, KG)                      │
│                   │                                                         │
│          ┌────────┴────────┬─────────────┐                                  │
│          ▼                 ▼             ▼                                  │
│      Semantic         Keyword          Knowledge Graph                      │
│      Results          Results          Results                              │
│          │                 │              │                                 │
│          └────────────┬────┴──────────────┘                                 │
│                       ▼                                                     │
│                  Fused Results                                              │
│                                                                             │
│  ─────────────────────────────────────────────────────────────              │
│                                                                             │
│  4. Continuous Learning RAG                                                 │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│                                                                             │
│  User feedback ──► Update embeddings ──► Improved retrieval                 │
│       │                                                                     │
│       │                                                                     │
│       └──────────────► Update retrieval model                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What This Means for Practitioners

- More sophisticated tools (LangGraph, AutoGen)
- Better evaluation frameworks
- Production-ready patterns emerging
- Cost optimization becoming critical

---

---

*Previous: [Why RAG](why-rag.md)*

*Next: [Classic RAG Architecture](../2-architectures/classic-rag.md)*
