# Production Deployment Guide

## Overview

This guide covers deploying RAG systems to production, including containerization, API development, orchestration, monitoring, and CI/CD pipelines.

## Deployment Architecture

```
Production RAG Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│                        Production Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│  │  Client  │────►│  API     │────►│   RAG    │────►│ Vector   │        │
│  │  Apps    │     │  Server  │     │  Engine  │     │  DB      │        │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘        │
│                         │                │                              │
│                         ▼                ▼                              │
│                   ┌──────────┐     ┌──────────┐                         │
│                   │  Load    │     │   LLM    │                         │
│                   │ Balancer │     │ Provider │                         │
│                   └──────────┘     └──────────┘                         │
│                                                                         │
│  ┌──────────┐     ┌────────────┐     ┌──────────┐                       │
│  │  Redis   │     │  Prometheus│     │ Grafana  │                       │
│  │  Cache   │     │ + Grafana  │     │          │                       │
│  └──────────┘     └────────────┘     └──────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Which Deployment Option Should You Use?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Deployment Options Decision Map                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Start                                                                    │
│      │                                                                      │
│      ▼                                                                      │
│    What's your use case?                                                    │
│      │                                                                      │
│      ├─ Development/Testing ──► Docker Compose (quick, easy)                │
│      │                                                                      │
│      ├─ Personal/Small Project ──► Docker + Redis (simple production)       │
│      │                                                                      │
│      ├─ Startup/SMB ──► Docker + Cloud (managed services)                   │
│      │                                                                      │
│      └─ Enterprise ──► Kubernetes (full scalability)                        │
│                                                                             │
│    Quick Comparison:                                                        │
│    ┌──────────────────┬─────────────┬─────────────┬─────────────┐           │
│    │                  │ Docker      │ Docker +    │ Kubernetes  │           │
│    │                  │ Compose     │ Cloud       │             │           │ 
│    ├──────────────────┼─────────────┼─────────────┼─────────────┤           │
│    │ Setup Time       │ Minutes     │ Hours       │ Days        │           │
│    │ Complexity       │ Low         │ Medium      │ High        │           │
│    │ Scalability      │ Single node │ Manual      │ Automatic   │           │
│    │ Cost             │ $ (server)  │ $$          │ $$$         │           │
│    │ Maintenance      │ Low         │ Medium      │ High        │           │
│    └──────────────────┴─────────────┴─────────────┴─────────────┘           │
│                                                                             │
│    When to choose what:                                                     │
│    • Docker Compose: Learning, testing, simple apps                         │
│    • Docker + Cloud: Production apps, startup                               │
│    • Kubernetes: Enterprise, auto-scaling, high availability                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Docker Containerization

**When to use Docker:** For local development, testing, or simple production deployments.

```dockerfile
# Use Python slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - ollama
      - redis
    volumes:
      - ./data:/app/data

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  ollama-data:
  redis-data:
```

## 2. FastAPI Application

**When to use FastAPI:** For creating a production REST API. FastAPI provides automatic docs, validation, and is async-native.

### Basic RAG API Server

```python
"""
Production RAG API Server
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import redis

app = FastAPI(
    title="RAG API",
    description="Production RAG system API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize components
llm = ChatOllama(model="llama3.2", base_url=OLLAMA_BASE_URL)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Redis cache
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool = False
    latency_ms: float

# Cache dependency
def get_cache():
    return redis_client

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, cache=Depends(get_cache)):
    """Query the RAG system."""
    
    import time
    start = time.time()
    
    # Check cache
    if request.use_cache:
        cached = cache.get(request.question)
        if cached:
            return QueryResponse(
                answer=cached["answer"],
                sources=cached["sources"],
                cached=True,
                latency_ms=(time.time() - start) * 1000
            )
    
    # Retrieve documents
    docs = retriever.invoke(request.question)
    context = "\n\n".join([d.page_content for d in docs])
    
    # Generate response
    prompt = f"""Context: {context}

Question: {request.question}

Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # Cache result
    if request.use_cache:
        cache.set(
            request.question,
            {"answer": answer, "sources": [d.page_content[:100] for d in docs]},
            ex=3600  # 1 hour TTL
        )
    
    latency_ms = (time.time() - start) * 1000
    
    return QueryResponse(
        answer=answer,
        sources=[d.page_content[:100] for d in docs],
        cached=False,
        latency_ms=latency_ms
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Implement custom metrics
    return {
        "total_requests": 0,
        "cache_hit_rate": 0.0,
        "avg_latency_ms": 0.0
    }
```

## 3. Kubernetes Deployment

**When to use Kubernetes:** For enterprise deployments requiring auto-scaling, high availability, or multi-service orchestration. Requires significant setup and maintenance.

### Kubernetes Deployment YAML

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OLLAMA_BASE_URL
          value: "http://ollama-service:11434"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        resources:
          requests:
            memory: "8Gi"
            gpu: "1"
          limits:
            memory: "16Gi"
            gpu: "1"
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

## 4. Monitoring & Observability

**When to use monitoring:** Always! Even in development, you need to understand latency, errors, and cache hit rates. Start simple, add more as you scale.

### Prometheus Metrics

```python
"""
Prometheus metrics for RAG
"""

from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
request_count = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['endpoint', 'status']
)

request_latency = Histogram(
    'rag_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

# Cache metrics
cache_hits = Counter('rag_cache_hits_total', 'Total cache hits')
cache_misses = Counter('rag_cache_misses_total', 'Total cache misses')

# LLM metrics
llm_token_count = Counter(
    'rag_llm_tokens_total',
    'Total LLM tokens used',
    ['model', 'type']
)

llm_latency = Histogram(
    'rag_llm_latency_seconds',
    'LLM inference latency'
)

# Vector store metrics
retrieval_latency = Histogram(
    'rag_retrieval_latency_seconds',
    'Vector retrieval latency'
)

retrieved_docs = Histogram(
    'rag_retrieved_documents',
    'Number of documents retrieved'
)

class MetricsMiddleware:
    """Middleware to track request metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            
            start_time = time.time()
            
            # Process request
            await self.app(scope, receive, send)
            
            # Record metrics
            latency = time.time() - start_time
            request_latency.labels(endpoint=path).observe(latency)
            request_count.labels(endpoint=path, status="success").inc()
        
        else:
            await self.app(scope, receive, send)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RAG System Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Latency P95",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P95 Latency"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rag_cache_hits_total / (rag_cache_hits_total + rag_cache_misses_total)",
            "legendFormat": "Hit Rate"
          }
        ]
      }
    ]
  }
}
```

## 5. CI/CD Pipeline

**When to use CI/CD:** For any project that will be updated. CI/CD ensures consistent builds, automated testing, and safe deployments.

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy RAG API

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Lint
      run: |
        pip install ruff
        ruff check src/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      uses: azure/k8s-set-context@v1
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy
      run: |
        kubectl set image deployment/rag-api \
        rag-api=ghcr.io/${{ github.repository }}:${{ github.sha }}
```

## 6. Security

**When to use security:** From day one! Even internal APIs need API keys and rate limiting to prevent abuse.

### API Key Management

```python
"""
Secure API key management
"""

from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
import os

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key from header."""
    
    valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
    
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return api_key

@app.post("/query")
async def protected_query(
    request: QueryRequest,
    api_key: str = Security(verify_api_key)
):
    """Protected endpoint requiring API key."""
    
    # Process request
    return await query(request)
```

### Rate Limiting

```python
"""
Rate limiting middleware
"""

from fastapi import Request, HTTPException
from typing import Dict
import time
from collections import defaultdict

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": requests_per_minute,
            "last_update": time.time()
        })
    
    async def check(self, request: Request) -> bool:
        """Check if request is allowed."""
        
        client_id = request.client.host
        bucket = self.buckets[client_id]
        
        # Refill tokens
        now = time.time()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + elapsed * (self.requests_per_minute / 60)
        )
        bucket["last_update"] = now
        
        # Check tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False

rate_limiter = RateLimiter(requests_per_minute=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting."""
    
    if not await rate_limiter.check(request):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return await call_next(request)
```

## 7. Performance Optimization

**When to use optimization:** Start monitoring first, then optimize only when needed. Don't prematurely optimize!

### Connection Pooling

```python
"""
Connection pool configuration
"""

from langchain_ollama import ChatOllama
import httpx

# Configure HTTP client with connection pooling
http_client = httpx.Client(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    http_client=http_client  # Reuse connections
)
```

### Batch Processing

```python
"""
Batch processing for high throughput
"""

from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchRAG:
    """RAG with batch processing."""
    
    def __init__(self, rag, batch_size: int = 10, max_workers: int = 5):
        self.rag = rag
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, questions: list) -> list:
        """Process multiple questions concurrently."""
        
        loop = asyncio.get_event_loop()
        
        # Submit all tasks
        futures = [
            loop.run_in_executor(self.executor, self.rag.query, q)
            for q in questions
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        return results
    
    def query_batch_sync(self, questions: list) -> list:
        """Synchronous batch query."""
        
        results = []
        
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            
            batch_results = self.executor.map(self.rag.query, batch)
            results.extend(batch_results)
        
        return results
```

## Quick Start Checklist

Follow this order to deploy your RAG system to production:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Production Deployment Checklist                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ☐ Step 1: Local Development                                                │
│     • Set up Docker Compose (this file: Docker Compose section)             │
│     • Test your RAG pipeline locally                                        │
│     • Verify all endpoints work                                             │
│                                                                             │
│  ☐ Step 2: Add API Layer                                                    │
│     • Implement FastAPI server (this file: FastAPI section)                 │
│     • Add request validation                                                │
│     • Test with curl or Postman                                             │
│                                                                             │
│  ☐ Step 3: Add Security                                                     │
│     • Implement API key authentication (this file: Security section)        │
│     • Add rate limiting                                                     │
│     • Test unauthorized access is blocked                                   │
│                                                                             │
│  ☐ Step 4: Add Monitoring                                                   │
│     • Add Prometheus metrics (this file: Monitoring section)                │
│     • Set up Grafana dashboard                                              │
│     • Verify metrics are being collected                                    │
│                                                                             │
│  ☐ Step 5: Set Up CI/CD                                                     │
│     • Add GitHub Actions workflow (this file: CI/CD section)                │
│     • Ensure tests run on every PR                                          │
│     • Set up automated Docker builds                                        │
│                                                                             │
│  ☐ Step 6: Deploy to Production                                             │
│     • Choose deployment option: Docker Compose vs Kubernetes                │
│     • Configure environment variables                                       │
│     • Set up monitoring alerts                                              │
│                                                                             │
│  ☐ Step 7: Optimize Performance                                             │
│     • Monitor latency and throughput                                        │
│     • Add caching if needed (this file: Performance section)                │
│     • Scale as needed                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Common Mistakes to Avoid

| Mistake | Why It's Bad | Fix |
|---------|--------------|-----|
| **No rate limiting** | API can be abused | Add rate limiter from Security section |
| **No monitoring** | Can't debug issues | Add Prometheus metrics |
| **Skipping tests** | Bugs in production | Add CI/CD pipeline |
| **No caching** | Slow + expensive | Add Redis cache |
| **Wrong chunk size** | Poor retrieval | Test different sizes |

---

## Production Considerations

### What Was Simplified (and Why)

This guide focuses on **understanding how RAG deployment works**. For learning purposes, several production requirements were intentionally simplified:

| Aspect | In This Guide | Why Simplified |
|--------|---------------|----------------|
| CORS | `allow_origins=["*"]` | Easy testing from any domain |
| Authentication | Simple API key header | Focus on core patterns |
| Input validation | No length/content limits | Focus on RAG logic |
| Secrets management | Environment variables | Focus on architecture |
| TLS/HTTPS | Not covered | Assumes reverse proxy handles it |
| Health checks | Basic `/health` only | Focus on deployment patterns |

### Security Essentials for Production

The patterns in this guide are correct. In production, you need to add security layers around them:

| Concept | Learning Version | Production Version |
|---------|------------------|---------------------|
| **CORS** | `allow_origins=["*"]` | Restrict to your specific domains |
| **Authentication** | Single hardcoded API key | OAuth2, JWT, or rotatable API keys |
| **Input validation** | No limits | Rate limiting + query length limits + sanitization |
| **Secrets** | Env vars in docker-compose | HashiCorp Vault, AWS Secrets Manager |
| **TLS/SSL** | Not covered | Terminate at load balancer (nginx, cloud LB) |
| **Audit logging** | Not covered | Log all requests with user IDs |

### Operational Essentials Checklist

Before deploying to production, ensure you have:

- [ ] **Health checks**: Add `/health/ready` (checks dependencies) and `/health/live` endpoints
- [ ] **Graceful shutdown**: Handle SIGTERM, drain connections, close pools
- [ ] **Structured logging**: JSON logs with request IDs for tracing
- [ ] **Vector DB backups**: Export/import strategy for your embeddings
- [ ] **Monitoring alerts**: Not just dashboards - alert on errors and high latency
- [ ] **Resource limits**: Set CPU/memory limits in Docker and Kubernetes
- [ ] **Retry logic**: Add exponential backoff for LLM API calls

### Key Takeaway

> The **patterns** in this guide are correct. The **implementation** is simplified for learning. In production, add security layers around these patterns without changing the core architecture.

---

*[Previous: Common Mistakes to Avoid] • [Next: Security Considerations](security-considerations.md)*

---

*For production hardening details, see [Production Hardening Guide](production-hardening.md)*
