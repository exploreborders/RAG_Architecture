# Production Hardening Guide

## Overview

This guide bridges the gap between the learning examples in our deployment documentation and production-ready deployments. It covers the security, infrastructure, and operational changes needed for production RAG systems.

> **Prerequisite**: Read [Production Deployment](production-deployment.md) first to understand the patterns.

## When to Use This Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Hardening Decision Map                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Have you deployed your RAG system?                                       │
│      │                                                                      │
│      ├─ No ──► Start with production-deployment.md first                    │
│      │                                                                      │
│      └─ Yes ─► What's your deployment context?                              │
│                   │                                                         │
│                   ├─ Internal API ──► Start here, add auth + monitoring     │
│                   │                                                         │
│                   ├─ Public API ──► Add all sections (auth, TLS, rate limit)│
│                   │                                                         │
│                   └─ Enterprise ──► Add all + compliance + audit logging    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. API Security Hardening

### CORS Configuration

The learning example allows all origins for easy testing. Production should be restricted:

```python
# ❌ Learning version (from production-deployment.md)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Production version
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)
```

### Input Validation

Add query validation to prevent abuse:

```python
"""
Production Input Validation
"""

from pydantic import BaseModel, Field
from typing import Optional
import re

class QueryRequest(BaseModel):
    question: str = Field(
        ...,  # Required
        min_length=1,      # Minimum 1 character
        max_length=2000,  # Maximum 2000 characters
        description="User question"
    )
    use_cache: bool = Field(default=True)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "use_cache": True
            }
        }

class QueryValidator:
    """Production query validator."""
    
    def __init__(self):
        self.max_length = 2000
        self.blocked_patterns = [
            r"<script",
            r"javascript:",
            r"on\w+\s*=",
        ]
        self.compiled = [re.compile(p, re.I) for p in self.blocked_patterns]
    
    def validate(self, query: str) -> tuple[bool, str | None]:
        """Validate query. Returns (is_valid, error_message)."""
        
        # Check length
        if len(query) > self.max_length:
            return False, f"Query exceeds {self.max_length} characters"
        
        # Check for blocked patterns
        for pattern in self.compiled:
            if pattern.search(query):
                return False, "Query contains blocked content"
        
        return True, None
```

> **Note**: For more advanced input validation (PII detection, prompt injection protection), see [Security Considerations](security-considerations.md).

### Rate Limiting (Enhanced)

The learning example uses simple IP-based rate limiting. Production should use API key-based limiting:

```python
"""
Production Rate Limiting
"""

from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
from collections import defaultdict
import time

api_key_header = APIKeyHeader(name="X-API-Key")

class ProductionRateLimiter:
    """Rate limiter based on API keys, not IPs.
    
    Note: This in-memory implementation works for single-instance deployments.
    For multi-replica deployments, use Redis-backed rate limiting (see below).
    """
    
    def __init__(self):
        # {api_key: {"tokens": float, "last_update": float}}
        self.buckets: dict[str, dict] = defaultdict(lambda: {
            "tokens": 60,
            "last_update": time.time()
        })
        
        # Rate limits per tier
        self.tiers = {
            "free": 10,      # 10 requests/minute
            "basic": 60,     # 60 requests/minute  
            "pro": 300,      # 300 requests/minute
            "enterprise": float("inf")
        }
    
    def get_tier(self, api_key: str) -> str:
        """Determine user's tier from API key."""
        # In production: lookup in database
        # This is simplified for learning
        if api_key.startswith("pro_"):
            return "pro"
        elif api_key.startswith("basic_"):
            return "basic"
        return "free"
    
    def check(self, api_key: str) -> bool:
        """Check if request is allowed."""
        
        tier = self.get_tier(api_key)
        rate = self.tiers.get(tier, 10)
        
        bucket = self.buckets[api_key]
        now = time.time()
        elapsed = now - bucket["last_update"]
        
        # Refill tokens
        bucket["tokens"] = min(
            rate,
            bucket["tokens"] + elapsed * (rate / 60)
        )
        bucket["last_update"] = now
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False

rate_limiter = ProductionRateLimiter()

@app.post("/query")
async def production_query(
    request: QueryRequest,
    api_key: str = Depends(api_key_header)
):
    """Production query endpoint with full security."""
    
    # Check rate limit
    if not rate_limiter.check(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Upgrade your plan."
        )
    
    # Validate input
    validator = QueryValidator()
    is_valid, error = validator.validate(request.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Process query...
```

### Redis-Backed Rate Limiting (For Multi-Replica)

For production deployments with multiple replicas, use Redis:

```python
"""
Redis-backed rate limiter for distributed deployments
"""

import redis

class RedisRateLimiter:
    """Rate limiter using Redis for shared state."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.tiers = {
            "free": 10,
            "basic": 60,
            "pro": 300,
            "enterprise": float("inf")
        }
    
    async def check(self, api_key: str, tier: str = "free") -> bool:
        """Check if request is allowed using Redis INCR."""
        
        rate = self.tiers.get(tier, 10)
        key = f"rate_limit:{api_key}"
        
        # Increment counter with expiry
        current = self.redis.incr(key)
        if current == 1:
            self.redis.expire(key, 60)  # 1 minute window
        
        return current <= rate
```
Or use a library like `slowapi` for easier integration.

### Secrets Management

Never store secrets in code or plain environment variables in production:

```python
# ❌ Learning version
REDIS_URL = "redis://localhost:6379"
API_KEY = "secret-key-123"

# ✅ Production version - use secrets management

# Option 1: Environment variables (minimum)
import os
REDIS_URL = os.getenv("REDIS_URL")  # Set in production env

# Option 2: pydantic-settings (recommended)
from pydantic_settings import BaseSettings  # v2
# from pydantic import BaseSettings  # v1

class Settings(BaseSettings):
    redis_url: str
    api_key: str
    ollama_base_url: str = "http://localhost:11434"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Option 3: HashiCorp Vault (enterprise)
# import hvac
# client = hvac.Client(url=os.getenv("VAULT_ADDR"))
# secrets = client.secrets.kv.v2.read_secret_path("rag-production")
```

---

## 2. Infrastructure Security

### TLS/SSL Configuration

Always use HTTPS in production. This is typically handled by a reverse proxy:

```yaml
# docker-compose.yml with TLS termination
services:
  nginx:
    image: nginx:latest
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - rag-api

  rag-api:
    expose:
      - "8000"
```

```nginx
# nginx.conf (simplified)
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    
    location / {
        proxy_pass http://rag-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### Network Policies (Kubernetes)

Restrict network communication between services:

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-api-policy
spec:
  podSelector:
    matchLabels:
      app: rag-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: ingress-controller
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    - to:
        - podSelector:
            matchLabels:
              app: ollama
      ports:
        - protocol: TCP
          port: 11434
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
```

---

## 3. Operational Security

### Enhanced Health Checks

Production should have separate liveness and readiness probes:

```python
"""
Production Health Checks

Note: Assumes these are defined elsewhere in your app:
- redis_client: Redis connection instance
- vectorstore: Your vector database (Chroma, etc.)
- os.getenv('OLLAMA_BASE_URL'): Your LLM endpoint
"""

from fastapi import FastAPI
import redis
import httpx
import os

# Note: These are assumed to be defined in your app:
# redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Simple liveness check - is the app running?
@app.get("/health/live")
async def liveness():
    return {"status": "alive"}

# Readiness check - can the app serve traffic?
@app.get("/health/ready")
async def readiness():
    checks = {}
    
    # Check Redis
    try:
        redis_client.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
    
    # Check Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.getenv('OLLAMA_BASE_URL')}/api/tags",
                timeout=5.0
            )
            checks["ollama"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        checks["ollama"] = f"unhealthy: {str(e)}"
    
# Check Vector DB (simple connectivity test)
    try:
        # Note: _collection is a private attribute; adjust for your vector DB
        # For Chroma: vectorstore._collection.count()
        # For other DBs: use appropriate health check method
        vectorstore._collection.count()
        checks["vectordb"] = "healthy"
    except Exception as e:
        checks["vectordb"] = f"unhealthy: {str(e)}"
    
    # Return unhealthy if any check failed
    all_healthy = all(v == "healthy" for v in checks.values())
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks
    }
```

### Request ID Tracing

Add request tracing for debugging:

```python
"""
Request ID Middleware
"""

import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import json

logger = logging.getLogger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add header to response
        response.headers["X-Request-ID"] = request_id
        
        return response

def log_with_request_id(logger, message: str, request_id: str, **kwargs):
    """Log with request ID context."""
    logger.info(
        json.dumps({
            "message": message,
            "request_id": request_id,
            **kwargs
        })
    )

# Usage in endpoints
@app.get("/query")
async def query_endpoint(request: Request):
    request_id = request.state.request_id
    log_with_request_id(logger, "Processing query", request_id)
    # ... process query
```

### Graceful Shutdown

Handle shutdowns properly to avoid dropped requests:

```python
"""
Graceful Shutdown Handler
"""

import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    yield
    
    # Shutdown
    logger.info("Shutting down gracefully...")
    
    # Stop accepting new requests
    logger.info("Stopping accepting new requests...")
    
    # Close connections
    await redis_client.close()
    await http_client.aclose()
    
    logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)
```

---

## 4. Monitoring & Alerting

### Alert Rules

Not just dashboards - actively alert on problems:

```yaml
# prometheus/alerts.yml
groups:
  - name: rag-alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(rag_requests_total{status="error"}[5m])) 
          / sum(rate(rag_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 5 seconds"
          
      - alert: LowCacheHitRate
        expr: |
          rag_cache_hits_total / (rag_cache_hits_total + rag_cache_misses_total) < 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 30%"
```

### SLO Definitions

Define Service Level Objectives:

| Metric | SLO | Description |
|--------|-----|-------------|
| Availability | 99.9% | Requests succeed |
| Latency (P95) | < 2s | 95% of requests under 2s |
| Latency (P99) | < 5s | 99% of requests under 5s |
| Error rate | < 0.1% | Less than 0.1% errors |
| Cache hit rate | > 60% | At least 60% cache hits |

---

## 5. Quick Checklist

Use this checklist before production deployment:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Production Hardening Checklist                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Security                                                                   │
│    ☐ CORS restricted to specific domains                                    │
│    ☐ Authentication implemented (API key, OAuth2, or JWT)                   │
│    ☐ Rate limiting enabled                                                  │
│    ☐ Input validation added                                                 │
│    ☐ TLS/SSL configured (HTTPS)                                             │
│    ☐ Secrets in vault (not env vars)                                        │
│                                                                             │
│  Infrastructure                                                             │
│    ☐ Network policies configured (K8s)                                      │
│    ☐ Database not exposed to public internet                                │
│    ☐ Resource limits set                                                    │
│                                                                             │
│  Operations                                                                 │
│    ☐ Health checks (liveness + readiness)                                   │
│    ☐ Graceful shutdown handler                                              │
│    ☐ Request ID tracing                                                     │
│    ☐ Structured JSON logging                                                │
│    ☐ Monitoring alerts configured                                           │
│    ☐ Backup strategy for vector DB                                          │
│                                                                             │
│  Testing                                                                    │
│    ☐ Load tested                                                            │
│    ☐ Security audit completed                                               │
│    ☐ Disaster recovery tested                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

| Topic | File | Description |
|-------|------|-------------|
| Deployment basics | [production-deployment.md](production-deployment.md) | How deployment works |
| RAG security | [security-considerations.md](security-considerations.md) | PII, injection, access control |
| Observability | [observability.md](observability.md) | Metrics and monitoring |
| This file | (current) | Production hardening guide |

---

## References

### Academic Papers

| Paper | Year | Focus |
|-------|------|-------|
| [PoisonedRAG: Knowledge Corruption Attacks](https://arxiv.org/abs/2402.07867) | 2024 | RAG security & attacks |
| [Prompt Injection Attacks](https://arxiv.org/abs/2306.05499) | 2023 | Injection prevention |

### Official Documentation

| Resource | Description |
|----------|-------------|
| [OWASP Top 10](https://owasp.org/www-project-top-ten/) | Web security |
| [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/) | API security |
| [LangChain Security](https://python.langchain.com/docs/security/) | Framework security |

### Blog Posts & Tutorials

| Blog | Description |
|------|-------------|
| [RAG Security Best Practices](https://www.pinecone.io/learn/rag-security) | Security patterns |
| [Securing RAG Pipelines](https://weaviate.io/blog/security-rag) | Implementation |
| [Rate Limiting Guide](https://cloud.google.com/architecture/rate-limiting) | API protection |

---

*Last updated: March 2026*
