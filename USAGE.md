# Multi-Hop RAG Usage Guide

This guide provides detailed instructions for using the multi-hop RAG system.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Indexing Documents](#indexing-documents)
4. [Querying](#querying)
5. [Advanced Usage](#advanced-usage)
6. [Production Deployment](#production-deployment)

## Installation

### Using UV (Recommended)

UV is a fast Python package installer and resolver. It's the recommended way to set up this project.

```bash
# Run the automated setup script
./setup_uv.sh

# Or manually:
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Configuration

### API Keys

You need two API keys:

1. **OpenAI API Key**: For GPT-4 access
   - Get it from: https://platform.openai.com/api-keys
   
2. **HuggingFace API Key**: For embedding generation
   - Get it from: https://huggingface.co/settings/tokens

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=sk-proj-your-actual-key-here
HUGGINGFACE_API_KEY=hf_your-actual-key-here
```

### Configuration Options

All configuration options can be set in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `HUGGINGFACE_API_KEY` | (required) | HuggingFace API key |
| `OPENAI_MODEL` | gpt-4-turbo-preview | OpenAI model to use |
| `HUGGINGFACE_EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `CHROMA_PERSIST_DIRECTORY` | ./chroma_db | ChromaDB storage location |
| `CHROMA_COLLECTION_NAME` | multi_hop_rag | Collection name |
| `CHUNK_SIZE` | 1000 | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `MAX_HOPS` | 3 | Maximum number of retrieval hops |
| `TOP_K_RETRIEVAL` | 5 | Number of documents per hop |
| `LOG_LEVEL` | INFO | Logging level |

## Indexing Documents

### Using Python API

```python
from multi_hop_rag import MultiHopRAGPipeline

pipeline = MultiHopRAGPipeline()

# Index a single document
pipeline.index_document("path/to/document.pdf")

# Index multiple documents
documents = ["doc1.pdf", "doc2.txt", "doc3.docx"]
pipeline.index_documents(documents)

# Index raw text
text = """
Title 17 - Commodity and Securities Exchanges
...
"""
pipeline.index_text(text, metadata={"source": "CFR"})
```

### Using CLI

```bash
# Index a single document
python examples/cli.py index --documents path/to/document.pdf

# Index multiple documents
python examples/cli.py index --documents doc1.pdf doc2.txt doc3.docx

# Check indexing status
python examples/cli.py stats
```

### Supported Document Types

- **PDF** (`.pdf`): Requires `pypdf` library (included)
- **Word** (`.doc`, `.docx`): Requires `python-docx` library (included)
- **Text** (`.txt`): Native support

### Best Practices for Indexing

1. **Hierarchical Documents**: The system is optimized for CFR-style documents with clear section markers (§, numbered sections, etc.)

2. **Document Metadata**: Add metadata to help with filtering:
```python
pipeline.index_document(
    "cfr_title_17.pdf",
    metadata={
        "title": "Title 17",
        "category": "securities",
        "year": "2024"
    }
)
```

3. **Batch Processing**: For large document sets, process in batches:
```python
import glob

pdf_files = glob.glob("documents/*.pdf")
for batch in range(0, len(pdf_files), 10):
    batch_files = pdf_files[batch:batch+10]
    pipeline.index_documents(batch_files)
```

## Querying

### Using Python API

```python
from multi_hop_rag import MultiHopRAGPipeline

pipeline = MultiHopRAGPipeline()

# Simple query
response = pipeline.query("What are the short sale requirements?")

print(response["answer"])
print(f"Retrieved {response['total_documents_retrieved']} documents")

# Access decomposed queries
for i, query in enumerate(response["decomposed_queries"], 1):
    print(f"Sub-query {i}: {query}")

# Access retrieved documents
for doc in response["retrieved_documents"][:5]:
    print(f"Section: {doc['metadata'].get('section')}")
    print(f"Content: {doc['content'][:100]}...")
```

### Using CLI

```bash
# Basic query
python examples/cli.py query --question "What are the locate requirements?"

# Verbose output (shows all details)
python examples/cli.py query -q "Explain Regulation SHO" --verbose
```

### Understanding the Response

The query response contains:

```python
{
    "question": "Original question",
    "answer": "Generated answer with citations",
    "decomposed_queries": [
        "Sub-query 1",
        "Sub-query 2",
        "Sub-query 3"
    ],
    "total_documents_retrieved": 15,
    "documents_by_hop": {
        "hop_1": 5,  # Documents from first hop
        "hop_2": 5,  # Documents from second hop
        "hop_3": 5   # Documents from third hop
    },
    "retrieved_documents": [
        {
            "id": "unique_id",
            "content": "document text",
            "metadata": {"section": "242.203", ...},
            "distance": 0.123  # Similarity score
        },
        ...
    ]
}
```

### Query Strategies

1. **Specific Questions**: Best for factual information
   - "What is the locate requirement for short sales?"
   - "When does the restricted period begin?"

2. **Complex Questions**: Leverages multi-hop retrieval
   - "How do short sale regulations relate to the restricted period?"
   - "What are the requirements and penalties for failure to close out?"

3. **Exploratory Questions**: Good for understanding topics
   - "Explain Regulation M"
   - "What is market manipulation prevention?"

## Advanced Usage

### Custom Configuration

```python
from multi_hop_rag import Settings, MultiHopRAGPipeline

# Create custom settings
settings = Settings(
    chunk_size=500,           # Smaller chunks
    chunk_overlap=100,        # Less overlap
    max_hops=4,              # More hops
    top_k_retrieval=10,      # More documents per hop
    openai_model="gpt-4"     # Different model
)

pipeline = MultiHopRAGPipeline(settings=settings)
```

### Using Individual Components

```python
from multi_hop_rag import (
    HierarchicalChunker,
    HuggingFaceEmbedder,
    ChromaStore,
    MultiHopRetriever
)

# Custom chunking
chunker = HierarchicalChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_text(your_text)

# Custom embedding
embedder = HuggingFaceEmbedder(
    api_key="your_key",
    model_name="sentence-transformers/all-mpnet-base-v2"
)
embeddings = embedder.embed_texts([chunk.content for chunk in chunks])

# Direct ChromaDB access
store = ChromaStore(persist_directory="./custom_db")
store.add_documents(chunks, embeddings)
```

### Async Queries

```python
import asyncio

async def main():
    pipeline = MultiHopRAGPipeline()
    response = await pipeline.aquery("Your question here")
    print(response["answer"])

asyncio.run(main())
```

## Production Deployment

### Environment Variables

Set environment variables directly instead of using .env file:

```bash
export OPENAI_API_KEY=your-key
export HUGGINGFACE_API_KEY=your-key
export LOG_LEVEL=WARNING
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN uv pip install --system .

# Run the application
CMD ["python", "your_app.py"]
```

### Performance Optimization

1. **Batch Indexing**: Process documents in batches
2. **Caching**: Enable HuggingFace model caching
3. **Connection Pooling**: Reuse OpenAI client connections
4. **Resource Limits**: Set appropriate chunk sizes and retrieval limits

### Monitoring

Enable detailed logging:

```python
from multi_hop_rag import setup_logging

setup_logging("DEBUG")
```

### Error Handling

```python
from multi_hop_rag import MultiHopRAGPipeline

pipeline = MultiHopRAGPipeline()

try:
    response = pipeline.query("Your question")
except Exception as e:
    print(f"Query failed: {e}")
    # Handle error (retry, log, etc.)
```

### Database Management

```python
# Get statistics
stats = pipeline.get_stats()
print(f"Total documents: {stats['total_documents']}")

# Reset database (careful!)
pipeline.reset()

# Backup ChromaDB
import shutil
shutil.copytree("./chroma_db", "./chroma_db_backup")
```

## Troubleshooting

### Issue: API Key Errors

```
ValueError: OPENAI_API_KEY must be set
```

**Solution**: Ensure .env file exists and contains valid keys

### Issue: Memory Errors

```
MemoryError: Unable to allocate array
```

**Solution**: Reduce chunk size or batch size:
```python
settings = Settings(chunk_size=500, top_k_retrieval=3)
```

### Issue: Slow Queries

**Solution**: 
- Reduce number of hops: `max_hops=2`
- Reduce documents per hop: `top_k_retrieval=3`
- Use faster embedding model

### Issue: Poor Answer Quality

**Solution**:
- Increase chunk size for more context
- Increase top_k_retrieval for more documents
- Use GPT-4 instead of GPT-3.5
- Improve document indexing with better metadata

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/joelchaconcastillo/multi-hop-rag/issues
- Documentation: README.md
