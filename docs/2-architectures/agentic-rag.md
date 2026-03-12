# Agentic RAG

## Overview

**Agentic RAG** represents the next evolution of RAG systems, where autonomous AI agents dynamically manage the retrieval and generation pipeline. Unlike classic RAG's linear flow, Agentic RAG uses agents that can plan, reflect, and adapt their retrieval strategies.

## Key Concepts

### What Makes It "Agentic"?

| Traditional RAG | Agentic RAG |
|-----------------|-------------|
| Fixed pipeline | Dynamic workflow |
| Single retrieval pass | Iterative refinement |
| No reflection | Self-critique & improvement |
| Predefined strategy | Adaptive strategy selection |
| No tool use | Multi-tool orchestration |

### Agent Design Patterns

```
Agentic RAG Design Patterns:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────┐
│                        Agentic Patterns                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Planning   │    │   Tool Use  │    │ Reflection  │              │
│  │             │    │             │    │             │              │
│  │ • Decompose │    │ • Search    │    │ • Evaluate  │              │
│  │   queries   │    │ • Retrieve  │    │ • Critique  │              │
│  │ • Create    │    │ • Generate  │    │ • Refine    │              │
│  │   execution │    │ • Compute   │    │             │              │
│  │   plan      │    │             │    │             │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                     │
│  ┌─────────────┐      ┌──────────────┐                              │
│  │   Memory    │      │  Multi Agent │                              │    
│  │             │      │ Collaboration│                              │
│  │ • Short-term│      │              │                              │
│  │ • Long-term │      │ • Specialist │                              │
│  │ • Context   │      │   agents     │                              │
│  │   window    │      │ • Debate     │                              │
│  └─────────────┘      └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

```
Agentic R## Architecture

AG Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        ┌─────────────────────┐
                        │     User Query      │
                        └──────────┬──────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │         Agent Brain          │
                    │  ┌────────────────────────┐  │
                    │  │    Planner Agent       │  │
                    │  │  - Decompose query     │  │
                    │  │  - Plan retrieval      │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Tool: Vector   │  │  Tool: Web      │  │  Tool: KG       │
    │  Search         │  │  Search         │  │  Traversal      │
    └────────┬────────┘  └─────────┬───────┘  └────────┬────────┘
             │                     │                   │
             └─────────────────────┼───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Reflection Agent        │
                    │  ┌────────────────────────┐  │
                    │  │ - Evaluate quality     │  │
                    │  │ - Check completeness   │  │
                    │  │ - Decide if re-retrieve│  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
                   Yes                             No
                    │                              │
                    ▼                              ▼
            ┌──────────────┐            ┌─────────────────────┐
            │ Re-plan/     │            │   Generator Agent   │
            │ Re-retrieve  │            │   (Final Answer)    │
            └──────────────┘            └─────────────────────┘
```

## Implementation with LangGraph

### Basic Agentic RAG

```python
"""
Agentic RAG with LangGraph
"""

from langgraph.graph import StateGraph, END
from langchain_community.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, TypedDict, Optional
import json

# Define state
class AgentState(TypedDict):
    question: str
    plan: List[str]
    retrieved_docs: List
    evaluation: str
    final_answer: Optional[str]
    iterations: int

# Initialize components (using Ollama - free, local)
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings
)

# Create tools
def vector_search(query: str) -> str:
    """Search vector database."""
    docs = vectorstore.similarity_search(query, k=4)
    return "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])

def web_search(query: str) -> str:
    """Search the web for information."""
    # Use Tavily or other search tool
    from langchain_community.tools import TavilySearchResults
    search = TavilySearchResults()
    return search.run(query)

# Define tools
tools = [
    Tool(
        name="vector_search",
        func=vector_search,
        description="Search internal knowledge base"
    ),
    Tool(
        name="web_search", 
        func=web_search,
        description="Search the web for current information"
    )
]

# Node 1: Planner
def planner_node(state: AgentState) -> AgentState:
    """Plan retrieval strategy."""
    
    prompt = f"""Given this question: {state['question']}
    
Create a plan to answer it. Consider:
1. What information is needed?
2. Which tools to use in what order?
3. How many retrieval steps?

Respond as a JSON list of steps."""
    
    response = llm.invoke(prompt)
    plan = json.loads(response)
    
    return {"plan": plan, "iterations": 0}

# Node 2: Retriever
def retriever_node(state: AgentState) -> AgentState:
    """Execute retrieval based on plan."""
    
    current_step = state["plan"][state["iterations"]]
    
    # Execute tool
    for tool in tools:
        if tool.name in current_step.lower():
            result = tool.run(current_step)
            break
    
    docs = vectorstore.similarity_search(state["question"], k=4)
    
    return {"retrieved_docs": docs}

# Node 3: Evaluator
def evaluator_node(state: AgentState) -> AgentState:
    """Evaluate retrieval quality."""
    
    prompt = f"""Question: {state['question']}

Retrieved context:
{state['retrieved_docs']}

Is this sufficient to answer the question? 
Consider:
- Are all aspects covered?
- Is the information accurate?
- Should we retrieve more?

Respond with:
- "sufficient" if ready to answer
- "need_more" if need additional retrieval
- "refine" if need to reformulate query"""

    evaluation = llm.invoke(prompt).content.strip().lower()
    
    return {"evaluation": evaluation}

# Node 4: Generator
def generator_node(state: AgentState) -> AgentState:
    """Generate final answer."""
    
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {state['question']}

Provide a detailed answer with citations."""
    
    answer = llm.invoke(prompt)
    
    return {"final_answer": answer}

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("evaluate", evaluator_node)
workflow.add_node("generate", generator_node)

# Define edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retrieve")
workflow.add_edge("retrieve", "evaluate")

# Conditional edge from evaluate
def should_generate(state: AgentState) -> str:
    if state["evaluation"] == "need_more":
        if state["iterations"] < 2:  # Max 3 iterations
            return "retry"
    return "generate"

workflow.add_conditional_edges(
    "evaluate",
    should_generate,
    {
        "retry": "planner",
        "generate": "generate"
    }
)

workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Usage
result = app.invoke({
    "question": "What are the latest developments in AI?",
    "plan": [],
    "retrieved_docs": [],
    "evaluation": "",
    "final_answer": None,
    "iterations": 0
})

print(result["final_answer"])
```

### Multi-Agent RAG with LangGraph

```python
"""
Multi-Agent RAG Architecture
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import List, TypedDict

# Define supervisor agent state
class MultiAgentState(TypedDict):
    question: str
    task_breakdown: List
    agent_results: dict
    synthesis: str

# Create specialized agents
research_agent = create_react_agent(
    llm,
    tools=[vector_search_tool, web_search_tool],
    prompt="You are a research agent. Find comprehensive information."
)

analysis_agent = create_react_agent(
    llm,
    tools=[analysis_tool],
    prompt="You are an analysis agent. Analyze and compare information."
)

synthesis_agent = create_react_agent(
    llm,
    tools=[],
    prompt="You are a synthesis agent. Combine findings into coherent answer."
)

# Supervisor node
def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """Supervisor decomposes question into tasks."""
    
    prompt = f"""Break down this question into subtasks:
{state['question']}

Each subtask should be handled by a different agent:
- research: Find information
- analysis: Analyze and compare
- synthesis: Combine findings

Return as JSON: {{"tasks": [{"agent": "research", "task": "..."}]}}"""
    
    tasks = json.loads(llm.invoke(prompt).content)
    return {"task_breakdown": tasks["tasks"]}

# Execute agent node
def execute_agent(agent, task: str):
    """Execute a single agent task."""
    result = agent.invoke({"messages": [("user", task)]})
    return result["messages"][-1].content

# Build workflow
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("research", lambda s: {"agent_results": {"research": execute_agent(research_agent, s["task_breakdown"][0]["task"])}})
graph.add_node("analysis", lambda s: {"agent_results": {"analysis": execute_agent(analysis_agent, s["agent_results"]["research"])}})
graph.add_node("synthesis", lambda s: {"synthesis": execute_agent(synthesis_agent, f"Combine: {s['agent_results']}")})

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "research")
graph.add_edge("research", "analysis")
graph.add_edge("analysis", "synthesis")
graph.add_edge("synthesis", END)

app = graph.compile()
```

## Implementation with LangGraph (ReAct Agent)

```python
"""
Agentic RAG using LangGraph
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import Tool

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent RAG agent. Your job is to:
1. Understand the question
2. Retrieve relevant information
3. Evaluate the retrieved context
4. Generate a comprehensive answer

You have access to:
- vector_search: Search the knowledge base
- web_search: Search for current information
- analyze: Analyze and compare results

Be thorough - don't hesitate to retrieve more information if needed."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent using LangGraph prebuilt
agent = create_react_agent(llm, tools)

# Run
result = agent.invoke({
    "messages": [("user", "What are the key differences between RAG architectures?")]
})
```

## Self-RAG Pattern

```python
"""
Self-RAG: Model reflects on its own retrieval
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import CommaSeparatedListOutputParser

class SelfRAGChain:
    """Self-Reflective RAG Chain."""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def run(self, question: str) -> str:
        # Step 1: Decide if retrieval is needed
        if_need_retrieval = self.llm.invoke(
            f"""Question: {question}
            
Should we retrieve additional context? Answer yes or no."""
        )
        
        if "yes" in if_need_retrieval.lower():
            # Step 2: Retrieve
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            context = ""
        
        # Step 3: Generate initial response
        initial_response = self.llm.invoke(
            f"""Context: {context}

Question: {question}

Provide your answer."""
        )
        
        # Step 4: Reflect on response
        reflection = self.llm.invoke(
            f"""Original question: {question}
Your answer: {initial_response}

Does this answer the question fully? What is missing?
Be critical."""
        )
        
        # Step 5: Refine if needed
        if "missing" in reflection.lower() or "incomplete" in reflection.lower():
            more_docs = self.retriever.invoke(reflection)
            more_context = context + "\n\n" + "\n\n".join([doc.page_content for doc in more_docs])
            
            final_response = self.llm.invoke(
                f"""Context: {more_context}

Question: {question}

Provide a refined answer."""
            )
        else:
            final_response = initial_response
        
        return final_response
```

## Pros and Cons

### ✅ Advantages

| Advantage | Description |
|-----------|-------------|
| **Adaptive** | Dynamic retrieval strategies |
| **Iterative** | Refines until quality achieved |
| **Multi-tool** | Can use multiple data sources |
| **Complex Tasks** | Handles multi-step reasoning |
| **Self-correction** | Evaluates and improves |

### ❌ Limitations

| Limitation | Description |
|------------|-------------|
| **Complex** | More components to manage |
| **Higher Cost** | Multiple LLM calls |
| **Latency** | Iterative process takes time |
| **Orchestration** | Requires careful design |
| **Debugging** | Harder to trace errors |

## When to Use Agentic RAG

### ✅ Best For

- Complex, multi-step questions
- When single retrieval isn't sufficient
- Dynamic data sources
- Research and analysis tasks
- When quality is critical

### ❌ Not Ideal For

- Simple, factual queries
- Real-time applications requiring low latency
- Cost-sensitive applications
- When classic RAG suffices

## Cost Optimization

```python
# Strategies to manage agentic RAG costs

class CostOptimizedAgent:
    def __init__(self, max_iterations=3, use_cache=True):
        self.max_iterations = max_iterations
        self.cache = {} if use_cache else None
    
    def run(self, question: str) -> str:
        # Check cache first
        if self.cache and question in self.cache:
            return self.cache[question]
        
        # ... execution logic ...
        
        # Cache result
        if self.cache:
            self.cache[question] = result
        
        return result

# Also consider:
# - Smaller models for simple decisions
# - Caching retrieved results
# - Limiting iterations
# - Early stopping criteria
```

---

## Try It Yourself

Practice implementing Agentic RAG with this notebook:

- [Agentic RAG Implementation Notebook](../notebooks/03-agentic-rag-implementation.ipynb)

---

*Next: [Multimodal RAG](../2-architectures/multimodal-rag/)*
