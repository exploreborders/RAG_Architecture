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

**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401))

Key innovations:
- First formal RAG architecture
- Combined dense retrieval (DPR - Dense Passage Retrieval) with seq2seq (sequence-to-sequence) generation
- Pre-trained on large-scale knowledge bases

### Architecture

```
Original RAG (2020):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query ──► Dense Retriever ──► Top-K Docs ──► Generator
              (DPR)             │            (BART/T5)
                                │
                         ┌──────┴──────┐
                         │  Wikipedia  │
                         │  Index      │
                         └─────────────┘
```

> **Summary (2020)**: The foundational RAG paper established the core pattern of retrieving relevant documents and using them as context for generation. This basic architecture became the baseline for all future improvements.

## 2021-2022: Advanced Retrieval

*Building on the foundational RAG architecture, researchers focused on improving the retrieval component itself.*

### Key Developments

| Year | Innovation | Description |
|------|------------|-------------|
| 2021 | DPR Improvements | Better dense retrievers |
| 2021 | BM25 + Dense | Hybrid search emergence (keyword + embedding) |
| 2022 | In-context RAG | Few-shot learning with retrieval |
| 2022 | Atlas | Fine-tuned retriever + generator |

### The Rise of Hybrid Search

Combining keyword and semantic search:
- **BM25**: Traditional keyword-based ranking, robust for exact matches
- **Dense (Embedding)**: Semantic understanding using neural networks
- **Hybrid**: Best of both worlds - combines keyword precision with semantic understanding

> **Summary (2021-2022)**: The focus shifted from basic retrieval to combining multiple search methods. Hybrid search emerged as a best practice, and researchers began exploring fine-tuned models specifically for retrieval tasks.

## 2023: Modular RAG

*As RAG matured, the community moved from a linear pipeline to a modular, composable architecture.*

### Paradigm Shift

RAG evolved from linear pipeline to **modular architecture**:

```
Modular RAG Components:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Input      │──►│  Retrieval  │──►│   Router    │
│  Processing │   │  Module     │   │   Module    │
└─────────────┘   └─────────────┘   └─────────────┘
                                           │
                   ┌─────────────┐         │
                   │   Fusion    │◄────────┤
                   │   Module    │         │
                   └─────────────┘   ┌─────────────┐
                          │          │   Memory    │
                          ▼          │   Module    │
                   ┌─────────────┐   └─────────────┘
                   │  Generator  │
                   │  Module     │
                   └─────────────┘
```

### Key Innovations (2023)

1. **Query Rewriting**: Improve query quality before retrieval
2. **Reranking**: Improve retrieved result quality using a second-stage scorer
3. **Chunking Strategies**: Optimize context windows
4. **Multi-vector**: Store multiple representations

> **Summary (2023)**: RAG became modular and composable. Query rewriting, reranking, and intelligent chunking became standard techniques. Developers could now mix and match components based on their use case.

## 2024-2025: Agentic & Advanced RAG

*The next leap came with autonomous systems that could decide when and how to retrieve information.*

### Self-RAG (2024)

**"Self-RAG: Learning to Retrieve, Generate, and Critique"** (Asai et al., 2024)

- Model learns to retrieve when needed
- Generates reflection tokens
- Self-critiques its outputs

> **Paper**: [arXiv:2410.13496](https://arxiv.org/abs/2410.13496)
> **GitHub**: [https://github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)

```
Self-RAG Process:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

> **Blog**: [GraphRAG: Unlocking LLM knowledge](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-knowledge/)
> **GitHub**: [microsoft/graphrag](https://github.com/microsoft/graphrag)

### Multimodal RAG (2025)

**Extending RAG beyond text**:
- Images, Video, Audio
- Cross-modal retrieval
- Unified embedding spaces

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

### Research Trends

| Trend                  | Description                            |
|------------------------|----------------------------------------|
| **Adaptive Retrieval** | When to retrieve, how much to retrieve |
| **Cost-Aware**         | Balance quality vs. computational cost |
| **Privacy-Preserving** | Secure retrieval in sensitive domains  |
| **Multi-modal**        | Unified understanding across formats   |
| **Agentic Control**    | Autonomous pipeline management         |

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

### What This Means for Practitioners

- More sophisticated tools (LangGraph, AutoGen)
- Better evaluation frameworks
- Production-ready patterns emerging
- Cost optimization becoming critical

---

## References

### Core RAG Papers

| Paper | Year | Link |
|-------|------|------|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.) | 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| Self-RAG: Learning to Retrieve, Generate, and Critique | 2024 | [arXiv:2410.13496](https://arxiv.org/abs/2410.13496) |
| A Systematic Survey of RAG | 2025 | [arXiv:2508.06401](https://arxiv.org/abs/2508.06401) |
| Comprehensive RAG Survey | 2025 | [arXiv:2506.00054](https://arxiv.org/abs/2506.00054) |
| Agentic RAG Survey | 2025 | [arXiv:2501.09136](https://arxiv.org/abs/2501.09136) |
| Multimodal RAG Survey | 2025 | [arXiv:2502.08826](https://arxiv.org/abs/2502.08826) |

### GraphRAG & Knowledge Graph RAG

| Resource | Description |
|----------|-------------|
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | Official Microsoft GraphRAG implementation |
| [GraphRAG Blog](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-knowledge/) | Microsoft Research blog post |

### Tools & Frameworks

| Tool | Description |
|------|-------------|
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Agentic RAG orchestration |
| [AutoGen](https://microsoft.github.io/autogen/) | Microsoft multi-agent framework |
| [LlamaIndex](https://www.llamaindex.ai/) | RAG-focused data framework |

---

*Next: [Classic RAG Architecture](../2-architectures/classic-rag/)*
