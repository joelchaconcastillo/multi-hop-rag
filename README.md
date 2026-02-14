# Multi-Hop RAG System

A production-ready multi-hop Retrieval-Augmented Generation (RAG) system designed for hierarchical document processing, built with ChromaDB, HuggingFace embeddings, OpenAI LLM, and LangGraph.

## Features

- **3-Hop Multi-Hop Retrieval**: Implements a sophisticated 3-hop retrieval strategy using LangGraph for complex document understanding
- **Hierarchical Document Processing**: Specialized chunking for CFR (Code of Federal Regulations) and other policy documents with hierarchical structure
- **Production-Ready**: Built with proper error handling, logging, and configuration management
- **Modern Stack**: 
  - ChromaDB for vector storage
  - HuggingFace API for embeddings
  - OpenAI GPT-4 for query decomposition and answer generation
  - LangGraph for orchestrating the multi-hop workflow
- **UV Compatible**: Fully compatible with UV package manager for fast, reliable Python package management

## Architecture

The system implements a 3-hop retrieval strategy:

1. **Hop 1**: Initial retrieval based on the decomposed query
2. **Hop 2**: Follow-up retrieval based on findings from Hop 1
3. **Hop 3**: Final retrieval to gather additional context
4. **Answer Generation**: Synthesize information from all hops using OpenAI LLM

```
User Query → Query Decomposition → Hop 1 Retrieval → Hop 2 Retrieval → Hop 3 Retrieval → Answer Generation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/joelchaconcastillo/multi-hop-rag.git
cd multi-hop-rag

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/joelchaconcastillo/multi-hop-rag.git
cd multi-hop-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required API Keys
OPENAI_API_KEY=your-openai-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here

# Model Configuration
OPENAI_MODEL=gpt-4-turbo-preview
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=multi_hop_rag

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration
MAX_HOPS=3
TOP_K_RETRIEVAL=5

# Logging
LOG_LEVEL=INFO
```

## Quick Start

### Basic Usage

```python
from multi_hop_rag import MultiHopRAGPipeline, setup_logging

# Setup logging
setup_logging("INFO")

# Initialize pipeline
pipeline = MultiHopRAGPipeline()

# Index a document
pipeline.index_document("path/to/cfr_document.pdf")

# Or index raw text
pipeline.index_text("""
Title 17 - Commodity and Securities Exchanges
Part 242 - Regulations M, SHO, NMS, AC, and MC
...
""")

# Query the system
response = pipeline.query(
    "What are the requirements for short sales?"
)

print(response["answer"])
```

### Using the CLI

```bash
# Index documents
python examples/cli.py index --documents docs/cfr_title_17.pdf docs/policy.txt

# Query the system
python examples/cli.py query --question "What are the locate requirements for short sales?"

# Query with verbose output
python examples/cli.py query -q "Explain regulation SHO" --verbose

# View statistics
python examples/cli.py stats

# Reset the database
python examples/cli.py reset
```

### Run the Example

```bash
python examples/basic_usage.py
```

## API Reference

### MultiHopRAGPipeline

Main pipeline class for the RAG system.

```python
from multi_hop_rag import MultiHopRAGPipeline

pipeline = MultiHopRAGPipeline()

# Index documents
pipeline.index_document("document.pdf", metadata={"category": "regulations"})
pipeline.index_text("text content", metadata={"source": "manual"})

# Query
response = pipeline.query("Your question here")

# Get statistics
stats = pipeline.get_stats()

# Reset database
pipeline.reset()
```

### Response Format

```python
{
    "question": "Original question",
    "answer": "Generated answer",
    "decomposed_queries": ["Sub-query 1", "Sub-query 2", ...],
    "total_documents_retrieved": 15,
    "documents_by_hop": {
        "hop_1": 5,
        "hop_2": 5,
        "hop_3": 5
    },
    "retrieved_documents": [
        {
            "id": "doc_id",
            "content": "document content",
            "metadata": {...},
            "distance": 0.123
        },
        ...
    ]
}
```

## Document Processing

### Supported Formats

- PDF (`.pdf`)
- DOCX (`.doc`, `.docx`)
- Plain Text (`.txt`)

### Hierarchical Structure

The system recognizes and preserves hierarchical structure in documents:

- CFR-style sections (e.g., §242.100)
- Numbered sections (e.g., 1.1, 1.1.1)
- Titled sections (Title, Chapter, Part, Subpart, Section)

### Example Document Structure

```
Title 17 - Commodity and Securities Exchanges
├── Part 242 - Regulations M, SHO, NMS, AC, and MC
│   ├── Subpart A - Regulation M
│   │   ├── §242.100 - Preliminary note
│   │   ├── §242.101 - Activities by distribution participants
│   │   └── §242.102 - Activities by issuers
│   └── Subpart B - Regulation SHO
│       ├── §242.200 - Definition of terms
│       └── §242.203 - Locate and delivery requirements
```

## Advanced Usage

### Custom Configuration

```python
from multi_hop_rag import Settings, MultiHopRAGPipeline

# Create custom settings
settings = Settings(
    chunk_size=500,
    chunk_overlap=100,
    max_hops=4,
    top_k_retrieval=10
)

# Initialize with custom settings
pipeline = MultiHopRAGPipeline(settings=settings)
```

### Using Individual Components

```python
from multi_hop_rag import (
    HierarchicalChunker,
    HuggingFaceEmbedder,
    ChromaStore,
    MultiHopRetriever,
    MultiHopRAGGraph
)

# Create individual components
chunker = HierarchicalChunker(chunk_size=1000)
embedder = HuggingFaceEmbedder(api_key="your-key")
store = ChromaStore(persist_directory="./db")
retriever = MultiHopRetriever(store, embedder)
rag_graph = MultiHopRAGGraph(retriever, settings)
```

## Development

### Install Development Dependencies

```bash
uv pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

## System Requirements

- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: Depends on document corpus size
- **API Access**: 
  - OpenAI API with GPT-4 access
  - HuggingFace API key

## Troubleshooting

### Common Issues

**Issue**: `OPENAI_API_KEY must be set`
- **Solution**: Ensure `.env` file exists and contains valid API keys

**Issue**: ChromaDB persistence errors
- **Solution**: Check write permissions for `CHROMA_PERSIST_DIRECTORY`

**Issue**: Embedding generation fails
- **Solution**: Verify HuggingFace API key and model name

**Issue**: Out of memory errors
- **Solution**: Reduce `CHUNK_SIZE` or `TOP_K_RETRIEVAL` values

## Performance Tuning

- **Chunk Size**: Larger chunks (1000-2000) for detailed documents, smaller (500-1000) for fragmented content
- **Top-K Retrieval**: Increase for broader context, decrease for focused answers
- **Max Hops**: 3 hops balance performance and thoroughness
- **Batch Size**: Adjust embedding batch size based on available memory

## License

See LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{multi_hop_rag,
  title={Multi-Hop RAG System},
  author={Joel Chacon Castillo},
  year={2024},
  url={https://github.com/joelchaconcastillo/multi-hop-rag}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.