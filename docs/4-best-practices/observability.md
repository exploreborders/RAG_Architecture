# Observability for RAG Systems

## Overview

Observability is critical for production RAG systems. It allows you to understand what's happening in your pipeline, debug issues, and continuously improve performance.

## Why Observability Matters

```
RAG Pipeline Without Observability:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query ──► Retrieve ──► Generate ──► Response
              │            │
              ▼            ▼
         "Why did we      "Why did the LLM
          miss this?       hallucinate?"

Without observability, you're flying blind!
```

## Core Observability Concepts

### Three Pillars

1. **Tracing**: Track the execution path through your pipeline
2. **Metrics**: Quantitative measurements (latency, cost, accuracy)
3. **Logging**: Detailed event records for debugging

## 1. Langfuse Integration

Langfuse is the most popular observability platform for LLM applications, including RAG systems.

### Setup

```python
"""
Langfuse Setup
"""

from langfuse import Langfuse

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
```

### Basic Tracing

```python
"""
Langfuse Tracing for RAG
"""

from langfuse import Langfuse, observe

@observe()
def retrieval_pipeline(query: str):
    """
    Complete RAG pipeline with Langfuse tracing.
    """
    
    # This function is now automatically traced!
    
    # Step 1: Retrieve documents
    docs = retriever.invoke(query)
    
    # Step 2: Generate context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 3: Generate response
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    
    return response

# Trace will appear in Langfuse dashboard automatically
result = retrieval_pipeline("What is RAG?")
```

### Manual Span Creation

```python
"""
Manual Langfuse Tracing
"""

from langfuse import Langfuse

langfuse = Langfuse()

def rag_pipeline(query: str):
    """RAG pipeline with manual tracing."""
    
    # Create a trace
    with langfuse.trace(name="rag_pipeline") as trace:
        
        # Add input
        trace.update(input={"query": query})
        
        # Step 1: Retrieval
        with trace.span(name="retrieval") as retrieval_span:
            start = time.time()
            docs = retriever.invoke(query)
            retrieval_span.update(
                output={"num_docs": len(docs)},
                metadata={
                    "latency_ms": (time.time() - start) * 1000,
                    "k": len(docs)
                }
            )
        
        # Step 2: Context preparation
        with trace.span(name="context_prep") as ctx_span:
            context = "\n\n".join([doc.page_content for doc in docs])
            ctx_span.update(metadata={"context_length": len(context)})
        
        # Step 3: Generation
        with trace.span(name="generation") as gen_span:
            start = time.time()
            response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
            gen_span.update(
                metadata={
                    "latency_ms": (time.time() - start) * 1000,
                    "model": "llama3.2"
                }
            )
        
        # Add output
        trace.update(output={"response": response.content})
        
        return response
```

### LangChain Integration

```python
"""
LangChain + Langfuse Integration
"""

from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

# Create handler
handler = CallbackHandler()

# Use with LangChain
llm = ChatOllama(
    model="llama3.2",
    callbacks=[handler]
)

# LangChain automatically traces
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.invoke("What is RAG?")
```

### Feedback Collection

```python
"""
User Feedback Collection
"""

# Collect feedback from users
def submit_feedback(trace_id: str, score: float, comment: str = None):
    """Submit user feedback to Langfuse."""
    
    langfuse.score(
        trace_id=trace_id,
        name="user_feedback",
        value=score,  # 0-1 or 1-5
        comment=comment
    )

# In your API endpoint
@app.post("/query")
async def query_with_feedback(request: QueryRequest):
    # trace_id is available from the trace object
    result = rag_pipeline(request.question)
    
    return {
        "answer": result.content,
        "trace_id": result.trace_id  # Get trace ID from result
    }

@app.post("/feedback")
async def submit_user_feedback(feedback: FeedbackRequest):
    submit_feedback(
        trace_id=feedback.trace_id,
        score=feedback.score,
        comment=feedback.comment
    )
    return {"status": "ok"}
```

### RAG Evaluation with Langfuse

```python
"""
RAG Evaluation in Langfuse using DeepEval
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import OllamaModel

def evaluate_rag_system(test_questions: list):
    """Evaluate RAG system using DeepEval."""
    
    # Initialize DeepEval with Ollama
    ollama_model = OllamaModel(model="llama3.2")
    faithfulness = FaithfulnessMetric(model=ollama_model)
    answer_relevancy = AnswerRelevancyMetric(model=ollama_model)
    context_precision = ContextualPrecisionMetric(model=ollama_model)
    context_recall = ContextualRecallMetric(model=ollama_model)
    
    # Generate predictions and evaluate
    results = []
    for q in test_questions:
        docs = retriever.invoke(q.question)
        context = "\n\n".join([d.page_content for d in docs])
        answer = llm.invoke(f"Context: {context}\n\nQuestion: {q.question}")
        
        # Create test case
        test_case = LLMTestCase(
            input=q.question,
            actual_output=answer.content,
            retrieval_context=[d.page_content for d in docs],
            expected_output=q.ground_truth
        )
        
        # Evaluate
        faithfulness.measure(test_case)
        answer_relevancy.measure(test_case)
        context_precision.measure(test_case)
        context_recall.measure(test_case)
        
        results.append({
            "question": q.question,
            "faithfulness": faithfulness.score,
            "answer_relevancy": answer_relevancy.score,
            "context_precision": context_precision.score,
            "context_recall": context_recall.score
        })
    
    # Log to Langfuse
    langfuse.score(
        name="faithfulness",
        value=results[0]["faithfulness"],
        trace_id=current_trace_id
    )
    
    return results
```

## 2. General Observability

### Prometheus Metrics

```python
"""
Prometheus Metrics for RAG
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Request metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['endpoint']
)

REQUEST_LATENCY = Histogram(
    'rag_request_duration_seconds',
    'Request latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Retrieval metrics
RETRIEVAL_COUNT = Histogram(
    'rag_retrieved_documents',
    'Number of documents retrieved',
    buckets=[1, 2, 3, 5, 10, 20]
)

RETRIEVAL_LATENCY = Histogram(
    'rag_retrieval_duration_seconds',
    'Retrieval latency'
)

# LLM metrics
LLM_TOKEN_COUNT = Counter(
    'rag_llm_tokens_total',
    'Total LLM tokens',
    ['model', 'type']  # type: prompt/completion
)

LLM_LATENCY = Histogram(
    'rag_llm_duration_seconds',
    'LLM inference latency'
)

# Cache metrics
CACHE_HITS = Counter('rag_cache_hits_total')
CACHE_MISSES = Counter('rag_cache_misses_total')

def rag_pipeline_with_metrics(query: str):
    """RAG pipeline with metrics collection."""
    
    REQUEST_COUNT.labels(endpoint='query').inc()
    
    with REQUEST_LATENCY.time():
        # Retrieval
        with RETRIEVAL_LATENCY.time():
            docs = retriever.invoke(query)
        
        RETRIEVAL_COUNT.observe(len(docs))
        
        # Generation
        context = "\n\n".join([d.page_content for d in docs])
        
        with LLM_LATENCY.time():
            response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
        
        LLM_TOKEN_COUNT.labels(model='llama3.2', type='prompt').inc(
            len(context.split())
        )
        LLM_TOKEN_COUNT.labels(model='llama3.2', type='completion').inc(
            len(response.content.split())
        )
    
    return response
```

### OpenTelemetry Integration

```python
"""
OpenTelemetry Tracing
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup OpenTelemetry
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Use in your code
@tracer.start_as_current_span("rag_pipeline")
def rag_pipeline(query: str):
    with tracer.start_as_current_span("retrieval") as span:
        span.set_attribute("query", query)
        docs = retriever.invoke(query)
        span.set_attribute("num_docs", len(docs))
    
    with tracer.start_as_current_span("generation") as span:
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    
    return response
```

## 3. Building Dashboards

### Key Metrics to Track

| Category | Metrics | Description |
|----------|---------|-------------|
| **Latency** | P50, P95, P99 | Response time distribution |
| **Throughput** | RPM, RPS | Requests per minute/second |
| **Retrieval** | Avg docs, precision | Quality of retrieval |
| **LLM** | Tokens, cost | Token usage and cost |
| **Cache** | Hit rate | Cache effectiveness |
| **Quality** | User feedback | User satisfaction |

### Grafana Dashboard Example

```json
{
  "dashboard": {
    "title": "RAG System Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {"expr": "rate(rag_requests_total[5m])", "legendFormat": "Requests/sec"}
        ]
      },
      {
        "title": "Latency P95",
        "type": "graph", 
        "targets": [
          {"expr": "histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))", "legendFormat": "P95"}
        ]
      },
      {
        "title": "Retrieval Quality",
        "type": "gauge",
        "targets": [
          {"expr": "rag_retrieved_documents_avg", "legendFormat": "Avg Docs"}
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {"expr": "rag_cache_hits_total / (rag_cache_hits_total + rag_cache_misses_total)"}
        ]
      }
    ]
  }
}
```

## 4. Debugging Common Issues

### Low Retrieval Quality

```python
"""
Debug Retrieval Issues
"""

def debug_retrieval(query: str, top_k: int = 10):
    """Debug retrieval to find issues."""
    
    # Get more results than usual
    docs = retriever.invoke(query)
    
    print(f"Query: {query}")
    print(f"\nRetrieved {len(docs)} documents:")
    
    for i, doc in enumerate(docs[:top_k], 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Content: {doc.page_content[:200]}...")
        
        # Calculate simple similarity
        query_terms = set(query.lower().split())
        doc_terms = set(doc.page_content.lower().split())
        overlap = query_terms & doc_terms
        print(f"   Term overlap: {overlap}")
```

### High Latency

```python
"""
Debug Latency Issues
"""

import time

def debug_latency(query: str):
    """Debug where latency is coming from."""
    
    times = {}
    
    # Retrieval
    start = time.time()
    docs = retriever.invoke(query)
    times['retrieval'] = time.time() - start
    
    # Context prep
    start = time.time()
    context = "\n\n".join([d.page_content for d in docs])
    times['context_prep'] = time.time() - start
    
    # Generation
    start = time.time()
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    times['generation'] = time.time() - start
    
    print("Latency breakdown:")
    for step, duration in times.items():
        print(f"  {step}: {duration*1000:.2f}ms")
    
    return times
```

### Hallucinations

```python
"""
Debug Hallucinations
"""

def check_hallucination(question: str, answer: str, docs: list):
    """Check if answer is grounded in context."""
    
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""Check if the answer is supported by the context.

Context: {context}

Answer: {answer}

Does the answer only use information from the context? Yes or no.
If no, which parts are not supported?"""
    
    response = llm.invoke(prompt)
    print(f"Hallucination check: {response.content}")
```

## 5. LangSmith (Alternative)

LangSmith is another popular option:

```python
"""
LangSmith Setup (alternative to Langfuse)
"""

from langsmith import Client

client = Client()

# Trace runs
run = client.create_run(
    project_name="my-rag-app",
    name="query",
    inputs={"query": "What is RAG?"}
)

# Add retrieval
client.create_run(
    parent_id=run.id,
    name="retrieval",
    outputs={"documents": [d.page_content for d in docs]}
)

# Add generation  
client.create_run(
    parent_id=run.id,
    name="generation",
    outputs={"response": response.content}
)
```

## Which Observability Tool Should You Use?

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    Observability Tools Decision Map                           │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│    Start                                                                      │
│      │                                                                        │
│      ▼                                                                        │
│    Do you need production-grade observability?                                │
│      │                                                                        │
│      ├─ No ──► Use basic debug functions (free!)                              │
│      │          • debug_retrieval()                                           │
│      │          • debug_latency()                                             │
│      │          • check_hallucination()                                       │
│      │                                                                        │
│      └─ Yes ──► Continue below                                                │
│                     │                                                         │
│                     ▼                                                         │
│              Are you using LangChain/LangGraph?                               │
│                     │                                                         │
│                     ├─ Yes ──► LangSmith ✓                                    │
│                     │          • Native integration                           │ 
│                     │          • Best LangChain support                       │
│                     │                                                         │
│                     └─ No ──► Continue below                                  │
│                                   │                                           │
│                                   ▼                                           │
│                        Do you want hosted (cloud)?                            │
│                                   │                                           │
│                                   ├─ Yes ──► Langfuse ✓                       │
│                                   │          • 10k free traces/month          │
│                                   │          • Works with any LLM             │
│                                   │          • Great UI/dashboard             │
│                                   │                                           │
│                                   └─ No ──► Continue below                    │
│                                                 │                             │
│                                                 ▼                             │
│                                      Do you want open source?                 │
│                                                 │                             │
│                                                 ├─ Yes ──► OpenTelemetry ✓    │
│                                                 │          • Industry standard│
│                                                 │          • Self-hosted      │
│                                                 │          • Most flexible    │
│                                                 │                             │
│                                                 └─ Combine tools!             │
│                                                    • Prometheus (metrics)     │
│                                                    • OpenTelemetry (traces)   │
│                                                    • Custom logging           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Quick Decision Guide

| Your Situation | Recommended | Why |
|---------------|------------|-----|
| **Quick debugging** | Debug functions | No setup needed |
| **LangChain + want easy** | LangSmith | Native integration |
| **Any LLM + hosted** | Langfuse | Best UI, free tier |
| **Enterprise + self-hosted** | OpenTelemetry | Full control |
| **Metrics + dashboards** | Prometheus + Grafana | Industry standard |
| **Production + full stack** | Langfuse + Prometheus | Best of both |

## Recommended Stack by Use Case

| Use Case | Recommended Stack |
|----------|------------------|
| **Development/Debugging** | Debug functions only |
| **Small project** | Langfuse (free tier) |
| **LangChain project** | LangSmith |
| **Enterprise** | OpenTelemetry + Prometheus + Grafana |
| **Best overall** | Langfuse + custom metrics |

---

## References

### Official Documentation

| Resource | Description |
|----------|-------------|
| [Langfuse](https://langfuse.com/docs) | RAG observability |
| [OpenTelemetry](https://opentelemetry.io/docs/) | Tracing standard |
| [Prometheus](https://prometheus.io/docs/) | Metrics collection |
| [Grafana](https://grafana.com/docs/) | Dashboards |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [RAG Observability](https://langfuse.com/blog/2025-10-28-rag-observability-and-evals) | RAG-specific guide |
| [Langfuse + LangChain](https://python.langchain.com/docs/integrations/providers/langfuse/) | LangChain integration |
| [OTel Python](https://opentelemetry.io/docs/instrumentation/python/) | Python SDK |

---

*Previous: [Production Hardening](production-hardening.md)*

*Next: [Security Considerations](security-considerations.md)*
