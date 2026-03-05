# Multimodal RAG

## Overview

**Multimodal RAG** extends traditional RAG to handle multiple modalities - text, images, audio, and video. This enables richer applications that can understand and reason across different types of content.

## Why Multimodal RAG?

### The Reality of Information

```
Information Modalities in Organizations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────────────────────────┐
│                     What We Have                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Text:      20%  - Documents, emails, reports                        │
│  Images:    30%  - Screenshots, diagrams, photos                     │
│  Video:     25%  - Meetings, tutorials, presentations                │
│  Audio:     15%  - Calls, podcasts, voice notes                      │
│  Tables:    10%  - Spreadsheets, databases                           │
│                                                                      │
│  Traditional RAG only handles: [=====>  Text <======]                │
│  Multimodal RAG handles:            [=====================> All      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Use Cases

| Domain | Application |
|--------|-------------|
| **Enterprise** | Search across all document types |
| **Education** | Course materials, lectures, slides |
| **Healthcare** | Medical images, clinical notes, scans |
| **E-commerce** | Product images, descriptions, reviews |
| **Media** | Video archives, podcasts, articles |

## Architecture

```
Multimodal RAG Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        ┌─────────────────────┐
                        │   User Query        │
                        │   (Any modality)    │
                        └──────────┬──────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │         Query Processing               │
              │  ┌──────────┐ ┌──────────┐ ┌───────┐   │
              │  │Text Embed│ │Image Emb │ │Audio  │   │
              │  └──────────┘ └──────────┘ │Embed  │   │
              │                            └───────┘   │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   Multimodal Index                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐      │
    │  │Text Vector  │  │Image Vector │  │Audio/Video      │      │
    │  │   Store     │  │   Store     │  │    Store        │      │
    │  └─────────────┘  └─────────────┘  └─────────────────┘      │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │      Knowledge Graph (Multimodal Entities)          │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │       Cross-Modal Retrieval            │
              │  - Text ↔ Image matching               │
              │  - Image ↔ Video matching              │
              │  - Audio ↔ Text alignment              │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │       Multi-Modal LLM Generation       │
              │  (e.g., GPT-4V, Claude 3, LLaVA)       │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────────┐
                        │   Multimodal        │
                        │   Response          │
                        └─────────────────────┘
```

## Implementation

### Basic Multimodal RAG with LangChain

```python
"""
Multimodal RAG Implementation
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredImageLoader,
    DirectoryLoader
)
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from PIL import Image
import os

class MultimodalRAG:
    """Handle multiple document modalities."""
    
    def __init__(self, persist_directory: str = "./multimodal_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.llm = ChatOllama(model="llama3.2")
    
    def load_documents(self, directory: str):
        """Load documents of various types."""
        
        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Load images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(DirectoryLoader(directory, glob=ext).load())
        
        # For images, we would use a multimodal embedding model
        # This is a simplified version
        
        return documents
    
    def process_images(self, image_paths: list) -> list:
        """Process images into descriptions."""
        
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        
        descriptions = []
        
        for img_path in image_paths:
            # Load image
            image = Image.open(img_path)
            
            # Use LLM to describe image
            # (In practice, use GPT-4V or similar)
            prompt = f"Describe this image in detail: {img_path}"
            
            # For actual multimodal, use:
            # response = llm.invoke([HumanMessage(
            #     content=[
            #         {"type": "text", "text": prompt},
            #         {"type": "image_url", "image_url": img_path}
            #     ]
            # )])
            
            # Simplified: store metadata
            descriptions.append({
                "path": img_path,
                "type": "image",
                "description": f"Image from {img_path}"
            })
        
        return descriptions
    
    def build_index(self, documents: list, image_descriptions: list):
        """Build multimodal vector index."""
        
        # Process text documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        texts = splitter.split_documents(documents)
        
        # Add image descriptions as documents
        from langchain_core.documents import Document
        
        image_docs = [
            Document(
                page_content=desc["description"],
                metadata={"source": desc["path"], "type": "image"}
            )
            for desc in image_descriptions
        ]
        
        all_docs = texts + image_docs
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def query(self, question: str) -> dict:
        """Query the multimodal RAG system."""
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=4)
        
        # Generate response
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return {
            "answer": answer,
            "sources": docs
        }
```

### Multimodal Embeddings

```python
"""
Using Multimodal Embedding Models
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformer

# Option 1: Use a multimodal embedding model
# like CLIP or BLIP

class MultimodalEmbeddings:
    """Embed text and images into same space."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        import torch
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def embed_text(self, text: str) -> list:
        """Embed text into vector."""
        inputs = self.processor(text=[text], return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings[0].tolist()
    
    def embed_image(self, image_path: str) -> list:
        """Embed image into vector."""
        from PIL import Image
        import torch
        
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings[0].tolist()
    
    def embed_documents(self, documents: list) -> list:
        """Embed mixed text/image documents."""
        embeddings = []
        
        for doc in documents:
            if doc.get("type") == "image":
                emb = self.embed_image(doc["path"])
            else:
                emb = self.embed_text(doc["content"])
            embeddings.append(emb)
        
        return embeddings
```

### Video RAG

```python
"""
Video RAG Implementation
"""

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VideoRAG:
    """RAG for video content."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
    
    def load_video(self, video_source: str, language: str = "en"):
        """Load video from YouTube or file."""
        
        if video_source.startswith("http"):
            # YouTube
            loader = YoutubeLoader.from_youtube_url(
                video_source,
                add_video_info=True
            )
        else:
            # Local video - would need video processing
            raise NotImplementedError("Local video not yet supported")
        
        docs = loader.load()
        
        # Add video metadata
        for doc in docs:
            doc.metadata["type"] = "video"
            doc.metadata["source"] = video_source
        
        return docs
    
    def build_index(self, documents: list):
        """Build video search index."""
        
        # Split into smaller chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        chunks = splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
    
    def query_video(self, question: str, threshold: float = 0.7):
        """Query video content."""
        
        results = self.vectorstore.similarity_search_with_score(
            question,
            k=4
        )
        
        # Filter by threshold
        filtered = [r for r, score in results if score < threshold]
        
        return filtered
```

### Audio RAG

```python
"""
Audio/Podcast RAG
"""

import whisper
from langchain_community.document_loaders import AudioTranscriptLoader

class AudioRAG:
    """RAG for audio content."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.whisper_model = whisper.load_model("base")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def load_audio(self, audio_path: str):
        """Load and transcribe audio."""
        
        # Transcribe
        transcript = self.transcribe_audio(audio_path)
        
        # Create document
        from langchain_core.documents import Document
        
        doc = Document(
            page_content=transcript,
            metadata={
                "source": audio_path,
                "type": "audio",
                "duration": self._get_duration(audio_path)
            }
        )
        
        return [doc]
    
    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        # Use audio library
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip()) if result.stdout else 0
    
    def build_index(self, documents: list):
        """Build audio search index."""
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        chunks = splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
```

## Multimodal LLM Integration

```python
"""
Using GPT-4V or Claude for Multimodal Generation
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class MultimodalGenerator:
    """Generate responses using multimodal LLM."""
    
    def __init__(self, model: str = "gpt-4-vision-preview"):
        self.llm = ChatOpenAI(model=model)
    
    def generate_with_images(self, question: str, images: list, context: str):
        """Generate answer incorporating images."""
        
        # Prepare messages
        content = [
            {"type": "text", "text": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        # Add images
        for img_path in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"}
            })
        
        messages = [HumanMessage(content=content)]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def generate_video_response(self, question: str, video_frames: list, transcript: str):
        """Generate answer from video frames and transcript."""
        
        content = [
            {"type": "text", "text": f"Transcript: {transcript}\n\nQuestion: {question}"}
        ]
        
        # Add key frames
        for frame in video_frames[:5]:  # Limit frames
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{frame}"}
            })
        
        messages = [HumanMessage(content=content)]
        
        response = self.llm.invoke(messages)
        return response.content
```

## Cross-Modal Retrieval

```python
"""
Cross-Modal Retrieval Examples
"""

# Text-to-Image Search
def text_to_image_search(query: str, image_dir: str) -> list:
    """Find images matching text query."""
    
    # Embed query
    query_embedding = multimodal_embeddings.embed_text(query)
    
    # Compare with image embeddings
    scores = []
    for img_file in os.listdir(image_dir):
        img_embedding = multimodal_embeddings.embed_image(
            os.path.join(image_dir, img_file)
        )
        score = cosine_similarity(query_embedding, img_embedding)
        scores.append((img_file, score))
    
    # Return top matches
    return sorted(scores, key=lambda x: x[1], reverse=True)[:5]

# Image-to-Text Search
def image_to_text_search(query_image: str, text_docs: list) -> list:
    """Find text documents matching image."""
    
    img_embedding = multimodal_embeddings.embed_image(query_image)
    
    scores = []
    for doc in text_docs:
        text_embedding = multimodal_embeddings.embed_text(doc.page_content)
        score = cosine_similarity(img_embedding, text_embedding)
        scores.append((doc, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:5]
```

## Tools & Frameworks

| Tool | Use Case |
|------|----------|
| **LLaVA** | Open-source multimodal LLM |
| **GPT-4V** | OpenAI multimodal model |
| **Claude 3** | Anthropic multimodal |
| **CLIP** | Cross-modal embeddings |
| **BLIP** | Image captioning & embeddings |
| **Whisper** | Audio transcription |
| **VidIQ** | Video understanding |

## Pros and Cons

### ✅ Advantages

| Advantage | Description |
|-----------|-------------|
| **Unified Search** | Search across all content types |
| **Rich Understanding** | Comprehend multiple modalities |
| **Better Context** | More complete information retrieval |
| **New Capabilities** | Enable previously impossible apps |

### ❌ Limitations

| Limitation | Description |
|------------|-------------|
| **Complex** | Multiple processing pipelines |
| **Expensive** | More compute for embeddings |
| **Alignment** | Cross-modal matching challenges |
| **Quality** | Depends on transcription/description quality |

---

*Next: [Advanced Patterns](./advanced-patterns.md)*
