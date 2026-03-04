#!/usr/bin/env python3
"""
Script to update all RAG documentation files to use Ollama as default.
"""

import os
import re

# Files to update (files with OpenAI imports that need Ollama)
FILES_TO_UPDATE = [
    "docs/4-best-practices/cost-optimization.md",
    "docs/4-best-practices/chunking-strategies.md",
    "docs/4-best-practices/query-optimization.md",
    "docs/4-best-practices/scaling-patterns.md",
    "docs/4-best-practices/security-considerations.md",
    "docs/3-technical/vector-databases.md",
    "docs/3-technical/evaluation-metrics.md",
    "docs/2-architectures/kg-rag.md",
    "docs/2-architectures/multimodal-rag.md",
    "docs/2-architectures/advanced-patterns.md",
]

# Import replacements
REPLACEMENTS = [
    # Embeddings
    (r'from langchain_community\.embeddings import OpenAIEmbeddings', 
     'from langchain_ollama import OllamaEmbeddings'),
    (r'from langchain_openai import OpenAIEmbeddings',
     'from langchain_ollama import OllamaEmbeddings'),
    (r'embedding=OpenAIEmbeddings\(\)',
     'embedding=OllamaEmbeddings(model="nomic-embed-text")'),
    (r'embedding=OpenAIEmbeddings\(model=',
     'embedding=OllamaEmbeddings(model='),
    (r'embedding_model: str = "text-embedding-3-small"',
     'embedding_model: str = "nomic-embed-text"'),
    
    # Chat models
    (r'from langchain_community\.chat_models import ChatOpenAI',
     'from langchain_ollama import ChatOllama'),
    (r'from langchain_openai import ChatOpenAI',
     'from langchain_ollama import ChatOllama'),
    (r'llm = ChatOpenAI\(model="gpt[^"]*"',
     'llm = ChatOllama(model="llama3.2")'),
    (r'llm=ChatOpenAI\(model="gpt',
     'llm=ChatOllama(model="llama3.2")'),
    (r'self\.llm = llm or ChatOpenAI\(model="gpt',
     'self.llm = llm or ChatOllama(model="llama3.2")'),
]

def update_file(filepath):
    """Update a single file with Ollama replacements."""
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} - not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [UPDATED] {filepath}")
        return True
    else:
        print(f"  [NO CHANGE] {filepath}")
        return False

def main():
    print("Updating RAG docs to use Ollama as default...\n")
    
    for filepath in FILES_TO_UPDATE:
        update_file(filepath)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
