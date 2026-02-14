"""Example usage of the multi-hop RAG system."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_hop_rag import MultiHopRAGPipeline, setup_logging


def main():
    """Main example function."""
    # Setup logging
    setup_logging("INFO")
    
    print("=" * 80)
    print("Multi-Hop RAG System Example")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = MultiHopRAGPipeline()
    
    # Print stats
    stats = pipeline.get_stats()
    print(f"\nPipeline Configuration:")
    print(f"  - Embedding Model: {stats['embedding_model']}")
    print(f"  - LLM Model: {stats['llm_model']}")
    print(f"  - Chunk Size: {stats['chunk_size']}")
    print(f"  - Max Hops: {stats['max_hops']}")
    print(f"  - Documents in DB: {stats['total_documents']}")
    
    # Example 1: Index sample text
    print("\n2. Indexing sample CFR-style document...")
    
    sample_text = """
Title 17 - Commodity and Securities Exchanges

Part 242 - Regulations M, SHO, NMS, AC, and MC and Customer Margin Requirements for Security Futures

Subpart A - Regulation M

§242.100 Preliminary note.

This regulation includes the following rules:
- Rule 101: Activities by distribution participants
- Rule 102: Activities by issuers and selling security holders
- Rule 103: Nasdaq passive market making
- Rule 104: Stabilizing and other activities in connection with an offering

§242.101 Activities by distribution participants.

Distribution participants are prohibited from bidding for or purchasing a covered security during the applicable restricted period. This rule is designed to prevent manipulative activity during securities distributions.

The restricted period begins on the later of one business day or five business days before the pricing of the security. For actively-traded securities, the restricted period is one business day, while for all other securities it is five business days.

§242.102 Activities by issuers and selling security holders.

This rule prohibits issuers, selling security holders, and their affiliated purchasers from bidding for or purchasing the security being distributed during the applicable restricted period. The purpose is to prevent artificial support of the security price during distribution.

Subpart B - Regulation SHO

§242.200 Definition of terms.

For purposes of this regulation, the following terms shall apply:

(a) "Short sale" means any sale of a security which the seller does not own or any sale which is consummated by the delivery of a security borrowed by, or for the account of, the seller.

(b) "Locate requirement" means that a broker-dealer must have reasonable grounds to believe that the security can be borrowed prior to effecting a short sale.

§242.203 Locate and delivery requirements.

A broker-dealer may not accept a short sale order unless the broker-dealer has:
1. Borrowed the security
2. Entered into a bona fide arrangement to borrow the security
3. Has reasonable grounds to believe that the security can be borrowed

This requirement is designed to address potential fails to deliver and naked short selling.
"""
    
    chunks_indexed = pipeline.index_text(
        sample_text,
        metadata={
            "source": "Sample CFR Document",
            "title": "Title 17 - Commodity and Securities Exchanges",
            "part": "242"
        }
    )
    print(f"  ✓ Indexed {chunks_indexed} chunks")
    
    # Example 2: Query the system
    print("\n3. Querying with multi-hop RAG...")
    
    question = "What are the requirements for short sales and how do they relate to the restricted period rules?"
    print(f"\nQuestion: {question}")
    
    response = pipeline.query(question)
    
    print(f"\nDecomposed Queries:")
    for i, query in enumerate(response["decomposed_queries"], 1):
        print(f"  {i}. {query}")
    
    print(f"\nRetrieved Documents:")
    print(f"  - Hop 1: {response['documents_by_hop']['hop_1']} documents")
    print(f"  - Hop 2: {response['documents_by_hop']['hop_2']} documents")
    print(f"  - Hop 3: {response['documents_by_hop']['hop_3']} documents")
    print(f"  - Total: {response['total_documents_retrieved']} documents")
    
    print(f"\nAnswer:")
    print("-" * 80)
    print(response["answer"])
    print("-" * 80)
    
    # Example 3: Show retrieved document snippets
    print("\n4. Retrieved Document Snippets:")
    for i, doc in enumerate(response["retrieved_documents"][:3], 1):
        section = doc["metadata"].get("section", "N/A")
        snippet = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
        print(f"\n  Document {i} (Section: {section}):")
        print(f"  {snippet}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
