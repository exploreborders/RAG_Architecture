#!/usr/bin/env python3
"""
Script to update notebooks to use Ollama.
"""

import json
import os

NOTEBOOKS = [
    "notebooks/01-classic-rag-implementation.ipynb",
    "notebooks/02-kg-rag-implementation.ipynb",
    "notebooks/03-agentic-rag-implementation.ipynb",
    "notebooks/04-evaluation-workshop.ipynb",
]

REPLACEMENTS = [
    ('"from langchain_community.embeddings import OpenAIEmbeddings"', 
     '"from langchain_ollama import OllamaEmbeddings"'),
    ('"from langchain_community.chat_models import ChatOpenAI"', 
     '"from langchain_ollama import ChatOllama"'),
    ('"embeddings = OpenAIEmbeddings()"', 
     '"embeddings = OllamaEmbeddings(model=\\"nomic-embed-text\\")"'),
    ('"llm = ChatOpenAI(model=\\"gpt-4\\", temperature=0)"',
     '"llm = ChatOllama(model=\\"llama3.2\\", temperature=0)"'),
    ('"llm = ChatOpenAI(model=\\"gpt-4\\")"',
     '"llm = ChatOllama(model=\\"llama3.2\\")"'),
    ('"llm = ChatOpenAI(model=\\"gpt-4o-mini\\")"',
     '"llm = ChatOllama(model=\\"llama3.2\\")"'),
    ('"self.llm = llm or ChatOpenAI(model=\\"gpt-4\\")"',
     '"self.llm = llm or ChatOllama(model=\\"llama3.2\\")"'),
]

def update_notebook(filepath):
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} - not found")
        return False
    
    with open(filepath, 'r') as f:
        nb = json.load(f)
    
    original = json.dumps(nb)
    
    for old, new in REPLACEMENTS:
        nb_str = json.dumps(nb)
        nb_str = nb_str.replace(old, new)
        nb = json.loads(nb_str)
    
    if json.dumps(nb) != original:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"  [UPDATED] {filepath}")
        return True
    else:
        print(f"  [NO CHANGE] {filepath}")
        return False

def main():
    print("Updating notebooks to use Ollama...\n")
    
    for nb in NOTEBOOKS:
        update_notebook(nb)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
