# Evaluation Metrics for RAG

## Overview

Evaluating RAG systems requires measuring both the **retrieval** quality and the **generation** quality. This document covers the key metrics and frameworks.

## Evaluation Framework Overview

```
RAG Evaluation Dimensions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG Evaluation                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐              ┌─────────────────────┐           │
│  │    RETRIEVAL        │              │    GENERATION       │           │
│  │    Evaluation       │              │    Evaluation       │           │
│  ├─────────────────────┤              ├─────────────────────┤           │
│  │ • Precision@K       │              │ • Faithfulness      │           │
│  │ • Recall@K          │              │ • Answer Relevance  │           │
│  │ • MRR               │              │ • Context Precision │           │
│  │ • Hit Rate          │              │ • Context Recall    │           │
│  │ • NDCG              │              │ • LLM-as-Judge      │           │
│  └─────────────────────┘              └─────────────────────┘           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     End-to-End Metrics                          │    │
│  │   • RAGAS  • ARES  • TruLens  • Custom Evaluations              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Retrieval Metrics

### Precision@K

**What it measures**: Fraction of retrieved documents that are relevant

```
Precision@K = (Number of relevant items in top K) / K
```

```python
def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate Precision@K."""
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant))
    return relevant_retrieved / k

# Example
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = ["doc1", "doc3", "doc5"]

precision_3 = precision_at_k(retrieved, relevant, 3)  # 2/3 = 0.67
precision_5 = precision_at_k(retrieved, relevant, 5)  # 3/5 = 0.60
```

### Recall@K

**What it measures**: Fraction of relevant documents that were retrieved

```
Recall@K = (Number of relevant items in top K) / (Total relevant documents)
```

```python
def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Calculate Recall@K."""
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant))
    return relevant_retrieved / len(relevant)

# Example
recall_3 = recall_at_k(retrieved, relevant, 3)  # 2/3 = 0.67
recall_5 = recall_at_k(retrieved, relevant, 5)  # 3/3 = 1.00
```

### Mean Reciprocal Rank (MRR)

**What it measures**: How highly the first relevant document is ranked

```
MRR = (1/N) * Σ (1/rank_i)

where rank_i is the position of first relevant document for query i
```

```python
def mean_reciprocal_rank(queries_results: list) -> float:
    """Calculate MRR."""
    rr_sum = 0
    
    for retrieved, relevant in queries_results:
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                rr_sum += 1 / rank
                break
    
    return rr_sum / len(queries_results)

# Example
queries_results = [
    (["doc1", "doc2", "doc3"], ["doc1"]),      # rank 1
    (["doc2", "doc1", "doc3"], ["doc1"]),      # rank 2
    (["doc3", "doc2", "doc1"], ["doc1"]),      # rank 3
]

mrr = mean_reciprocal_rank(queries_results)  # (1/1 + 1/2 + 1/3)/3 = 0.61
```

### Hit Rate

**What it measures**: Whether at least one relevant document is in top K

```python
def hit_rate(retrieved: list, relevant: list, k: int) -> float:
    """Calculate Hit Rate@K."""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return 1.0 if len(retrieved_k & relevant_set) > 0 else 0.0
```

## Generation Metrics

### Using DeepEval

DeepEval is an open-source evaluation framework that supports Ollama for local evaluation (no API key required). It's great for testing RAG systems without relying on external APIs.

```python
"""
DeepEval Evaluation (supports Ollama for local evaluation)
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.models import OllamaModel

# Initialize with Ollama (no API key required)
ollama_model = OllamaModel(model="llama3.2")

# Create test case
test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG stands for Retrieval-Augmented Generation...",
    retrieval_context=["RAG is a framework...", "It combines retrieval..."],
    expected_output="RAG is Retrieval-Augmented Generation, a technique..."
)

# Initialize metrics
faithfulness = FaithfulnessMetric(model=ollama_model)
answer_relevancy = AnswerRelevancyMetric(model=ollama_model)
context_precision = ContextualPrecisionMetric(model=ollama_model)
context_recall = ContextualRecallMetric(model=ollama_model)

# Run evaluation
faithfulness.measure(test_case)
answer_relevancy.measure(test_case)
context_precision.measure(test_case)
context_recall.measure(test_case)

print(f"Faithfulness: {faithfulness.score}")
print(f"Answer Relevancy: {answer_relevancy.score}")
print(f"Context Precision: {context_precision.score}")
print(f"Context Recall: {context_recall.score}")
```

### Metric Definitions

| Metric | Measures | Ideal Score |
|--------|----------|-------------|
| **Faithfulness** | Does answer match retrieved context? | 1.0 |
| **Answer Relevancy** | How relevant is the answer to the question? | 1.0 |
| **Context Precision** | Are high-ranked chunks relevant? | 1.0 |
| **Context Recall** | Does retrieved context contain answer? | 1.0 |

### Custom Evaluation with LLM-as-Judge

LLM-as-Judge uses an LLM to evaluate RAG outputs. This approach is flexible and can evaluate semantic quality that traditional metrics miss.

**When to use LLM-as-Judge:** Use this when you want flexible, semantic evaluation without setting up external frameworks. It's especially useful for domain-specific evaluation where standard metrics don't apply.

```python
"""
LLM-as-Judge for RAG Evaluation
"""

from langchain_ollama import ChatOllama

class RAGEvaluator:
    """Evaluate RAG using LLM as judge."""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3.2")
    
    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> float:
        """Evaluate if answer is grounded in context."""
        
        prompt = f"""Evaluate whether the answer is faithful to the context.

Question: {question}

Context: {context}

Answer: {answer}

Rate faithfulness from 0-10 where:
- 10: Answer is fully supported by context
- 5: Answer partially supported
- 0: Answer contradicts or not supported by context

Respond with only the number."""

        response = self.llm.invoke(prompt)
        return float(response.strip()) / 10
    
    def evaluate_answer_quality(self, question: str, answer: str) -> float:
        """Evaluate answer relevance and completeness."""
        
        prompt = f"""Evaluate the quality of this answer.

Question: {question}

Answer: {answer}

Rate from 0-10:
- 10: Complete, accurate, well-structured
- 5: Partial or somewhat relevant
- 0: Irrelevant or wrong

Respond with only the number."""

        response = self.llm.invoke(prompt)
        return float(response.strip()) / 10
    
    def evaluate_context_quality(self, question: str, contexts: list) -> float:
        """Evaluate retrieval quality."""
        
        prompt = f"""Evaluate how relevant the retrieved contexts are to the question.

Question: {question}

Contexts:
{chr(10).join([f'{i+1}. {c}' for i, c in enumerate(contexts)])}

Rate from 0-10:
- 10: All contexts highly relevant
- 5: Some relevant
- 0: None relevant

Respond with only the number."""

        response = self.llm.invoke(prompt)
        return float(response.strip()) / 10
    
    def full_evaluation(self, question: str, answer: str, contexts: list) -> dict:
        """Run full evaluation suite."""
        
        return {
            "faithfulness": self.evaluate_faithfulness(question, answer, "\n\n".join(contexts)),
            "answer_quality": self.evaluate_answer_quality(question, answer),
            "context_quality": self.evaluate_context_quality(question, contexts),
            "avg_score": sum([
                self.evaluate_faithfulness(question, answer, "\n\n".join(contexts)),
                self.evaluate_answer_quality(question, answer),
                self.evaluate_context_quality(question, contexts)
            ]) / 3
        }

# Usage
evaluator = RAGEvaluator()

result = evaluator.full_evaluation(
    question="What is RAG?",
    answer="RAG combines retrieval with generation...",
    contexts=["RAG is...", "It uses retrieval..."]
)

print(result)
```

### Traditional Generation Metrics

BLEU and ROUGE are traditional text generation metrics. Note: ROUGE is less useful for RAG systems since it measures overlap with reference text, but can be useful for evaluating summarization.

```python
"""
BLEU, ROUGE for text generation
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score."""
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

def calculate_rouge(reference: str, hypothesis: str) -> dict:
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

# Example
reference = "The cat sat on the mat"
hypothesis = "A cat was sitting on the mat"

bleu = calculate_bleu(reference, hypothesis)
rouge = calculate_rouge(reference, hypothesis)

print(f"BLEU: {bleu}")
print(f"ROUGE: {rouge}")
```

## TruLens Evaluation

TruLens is an evaluation framework from TruEra that provides detailed feedback on RAG quality. It's particularly useful for understanding *why* your RAG system is performing well or poorly.

**When to use TruLens:** Use TruLens when you need detailed instrumentation and feedback on your RAG pipeline, especially for debugging and optimization.

```python
"""
TruLens for RAG Evaluation
"""

from trulens.core import Feedback
from trulens.feedback import Groundedness
from trulens.providers.openai import OpenAI as TruLensOpenAI

# Setup
provider = TruLensOpenAI()
groundedness = Groundedness(provider=provider)

# Define feedback functions
f_groundedness = Feedback(
    groundedness.groundedness_measure,
    name="Groundedness"
).on(context.collect()).on_output()

f_answer_quality = Feedback(
    provider.quality,
    name="Answer Quality"
).on(question).on_output()

# Instrument and evaluate
from trulens.apps.rag import TruRag

app = TruRag(
    rag_chain=qa_chain,
    feedbacks=[f_groundedness, f_answer_quality]
)

# Evaluate
result = app.evaluate(
    question="What is RAG?",
    expected_answer="RAG is..."
)

print(result)
```

## End-to-End Benchmarking

```python
"""
Comprehensive RAG Benchmarking
"""

import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    query: str
    latency_ms: float
    retrieval_metrics: dict
    generation_metrics: dict
    overall_score: float

class RAGBenchmark:
    """Benchmark RAG system."""
    
    def __init__(self, rag_pipeline, evaluator):
        self.rag = rag_pipeline
        self.evaluator = evaluator
    
    def run(self, test_queries: List[dict]) -> List[BenchmarkResult]:
        """Run benchmark suite."""
        
        results = []
        
        for query_data in test_queries:
            # Time the query
            start = time.time()
            response = self.rag.invoke(query_data["question"])
            latency = (time.time() - start) * 1000
            
            # Evaluate
            eval_result = self.evaluator.full_evaluation(
                question=query_data["question"],
                answer=response["answer"],
                contexts=response["contexts"]
            )
            
            results.append(BenchmarkResult(
                query=query_data["question"],
                latency_ms=latency,
                retrieval_metrics={"precision": eval_result["context_quality"]},
                generation_metrics={
                    "faithfulness": eval_result["faithfulness"],
                    "answer_quality": eval_result["answer_quality"]
                },
                overall_score=eval_result["avg_score"]
            ))
        
        return results
    
    def summary(self, results: List[BenchmarkResult]) -> dict:
        """Generate summary statistics."""
        
        return {
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
            "avg_overall_score": sum(r.overall_score for r in results) / len(results),
            "avg_faithfulness": sum(r.generation_metrics["faithfulness"] for r in results) / len(results),
            "avg_context_quality": sum(r.retrieval_metrics["precision"] for r in results) / len(results),
        }
```

## Key Metrics Summary

| Category | Metric | When to Use |
|----------|--------|-------------|
| **Retrieval** | Precision@K | When false positives costly |
| **Retrieval** | Recall@K | When missing relevant docs costly |
| **Retrieval** | MRR | Ranking quality matters |
| **Generation** | Faithfulness | Grounding in context |
| **Generation** | Answer Relevancy | Response quality |
| **End-to-End** | RAGAS Score | Overall system performance |
| **End-to-End** | LLM-as-Judge | Flexible, semantic evaluation |

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [RAGAS: Evaluation Framework](https://arxiv.org/abs/2309.15296) | 2023 | RAG evaluation metrics |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG evaluation framework |
| [TruLens](https://truera.com/trulens/) | RAG evaluation |
| [ARES](https://github.com/stanford-futuredata/ARES) | Automated evaluation |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [RAG Evaluation](https://www.pinecone.io/learn/rag-evaluation) | Evaluation guide |
| [LLM-as-Judge](https://docs.wandb.ai/guides/reports/using-reports) | Judge-based evaluation |

---

*Previous: [Providers](providers.md)*

*Next: [Chunking Strategies](../4-best-practices/chunking-strategies.md)*
