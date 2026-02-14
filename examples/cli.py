"""Script to index documents and run queries."""

import sys
import argparse
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_hop_rag import MultiHopRAGPipeline, setup_logging


def index_documents(pipeline: MultiHopRAGPipeline, document_paths: list[str]) -> None:
    """Index documents from file paths."""
    print(f"\nIndexing {len(document_paths)} document(s)...")
    
    total_chunks = 0
    for doc_path in document_paths:
        path = Path(doc_path)
        if not path.exists():
            print(f"  ✗ File not found: {doc_path}")
            continue
        
        try:
            chunks = pipeline.index_document(str(path))
            total_chunks += chunks
            print(f"  ✓ {path.name}: {chunks} chunks indexed")
        except Exception as e:
            print(f"  ✗ {path.name}: Failed - {str(e)}")
    
    print(f"\nTotal: {total_chunks} chunks indexed")


def run_query(pipeline: MultiHopRAGPipeline, question: str, verbose: bool = False) -> None:
    """Run a query through the RAG system."""
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    response = pipeline.query(question)
    
    if verbose:
        print(f"\nDecomposed Queries:")
        for i, query in enumerate(response["decomposed_queries"], 1):
            print(f"  {i}. {query}")
        
        print(f"\nRetrieved Documents by Hop:")
        print(f"  - Hop 1: {response['documents_by_hop']['hop_1']} documents")
        print(f"  - Hop 2: {response['documents_by_hop']['hop_2']} documents")
        print(f"  - Hop 3: {response['documents_by_hop']['hop_3']} documents")
        print(f"  - Total: {response['total_documents_retrieved']} documents")
    
    print(f"\nAnswer:")
    print(response["answer"])
    print("-" * 80)
    
    if verbose:
        print(f"\nTop Retrieved Documents:")
        for i, doc in enumerate(response["retrieved_documents"][:3], 1):
            section = doc["metadata"].get("section", "N/A")
            snippet = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            print(f"\n  [{i}] Section: {section}")
            print(f"      {snippet}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Multi-hop RAG system for hierarchical document processing"
    )
    
    parser.add_argument(
        "command",
        choices=["index", "query", "stats", "reset"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--documents",
        "-d",
        nargs="+",
        help="Document file paths to index"
    )
    
    parser.add_argument(
        "--question",
        "-q",
        help="Question to query"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Initialize pipeline
    print("Initializing Multi-hop RAG Pipeline...")
    pipeline = MultiHopRAGPipeline()
    print("✓ Pipeline initialized")
    
    # Execute command
    if args.command == "index":
        if not args.documents:
            print("Error: --documents required for index command")
            sys.exit(1)
        index_documents(pipeline, args.documents)
    
    elif args.command == "query":
        if not args.question:
            print("Error: --question required for query command")
            sys.exit(1)
        run_query(pipeline, args.question, args.verbose)
    
    elif args.command == "stats":
        stats = pipeline.get_stats()
        print("\nPipeline Statistics:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Embedding Model: {stats['embedding_model']}")
        print(f"  LLM Model: {stats['llm_model']}")
        print(f"  Chunk Size: {stats['chunk_size']}")
        print(f"  Chunk Overlap: {stats['chunk_overlap']}")
        print(f"  Max Hops: {stats['max_hops']}")
        print(f"  Top-K Retrieval: {stats['top_k_retrieval']}")
    
    elif args.command == "reset":
        confirm = input("Are you sure you want to reset the vector store? (yes/no): ")
        if confirm.lower() == "yes":
            pipeline.reset()
            print("✓ Vector store reset successfully")
        else:
            print("Reset cancelled")


if __name__ == "__main__":
    main()
