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
- Combined dense retrieval (DPR) with seq2seq generation
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

## 2021-2022: Advanced Retrieval

### Key Developments

| Year | Innovation | Description |
|------|------------|-------------|
| 2021 | DPR Improvements | Better dense retrievers |
| 2021 | BM25 + Dense | Hybrid search emergence |
| 2022 | In-context RAG | Few-shot learning with retrieval |
| 2022 | Atlas | Fine-tuned retriever + generator |

### The Rise of Hybrid Search

Combining keyword and semantic search:
- **BM25**: Traditional, robust for exact matches
- **Dense (Embedding)**: Semantic understanding
- **Hybrid**: Best of both worlds

## 2023: Modular RAG

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
2. **Re-ranking**: Improve retrieved result quality
3. **Chunking Strategies**: Optimize context windows
4. **Multi-vector**: Store multiple representations

## 2024-2025: Agentic & Advanced RAG

### Self-RAG (2024)

**"Self-RAG: Learning to Retrieve, Generate, and Critique"**

- Model learns to retrieve when needed
- Generates reflection tokens
- Self-critiques its outputs

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

**"From Local to Global: A Graph RAG Approach"**

- Builds knowledge graphs from documents
- Community summarization
- Global knowledge synthesis

### Multimodal RAG (2025)

**Extending RAG beyond text**:
- Images, Video, Audio
- Cross-modal retrieval
- Unified embedding spaces

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

*Next: [Classic RAG Architecture](../2-architectures/classic-rag/)*
