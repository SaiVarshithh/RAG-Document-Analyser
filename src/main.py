"""
Command-line interface for the RAG system.
"""
import argparse
import os
from app import RAGSystem
from config.config import config
from pprint import pprint

def main():
    """Main function to handle CLI commands."""
    parser = argparse.ArgumentParser(description="RAG System for Technical Documentation")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index documents from a directory.")
    ingest_parser.add_argument(
        "--path",
        type=str,
        default=config.documents_dir,
        help=f"Path to the documents directory (default: {config.documents_dir})"
    )

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question to the RAG system.")
    ask_parser.add_argument("query", type=str, help="The question to ask.")
    ask_parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve (default: 5)")

    args = parser.parse_args()

    # Initialize the RAG system
    rag_system = RAGSystem()

    if args.command == "ingest":
        if not os.path.exists(args.path):
            print(f"Error: The specified path does not exist: {args.path}")
            # Create a sample file for the user
            os.makedirs(config.documents_dir, exist_ok=True)
            sample_file_path = os.path.join(config.documents_dir, "sample_document.txt")
            with open(sample_file_path, "w") as f:
                f.write("This is a sample document. RAG systems use retrieval to augment generation.")
            print(f"A sample documents directory and a sample file have been created at: '{config.documents_dir}'.")
            print("Please add your documents to this directory and run the ingest command again.")
            return
        
        print(f"Starting ingestion from '{args.path}'...")
        rag_system.setup_pipeline(args.path)
        print("Ingestion complete. The index has been created.")

    elif args.command == "ask":
        print(f"Asking question: '{args.query}'")
        response = rag_system.ask_question(args.query, top_k=args.top_k)
        
        print("\n--- Answer ---")
        print(response['answer'])
        print("\n--- Sources ---")
        if response['sources']:
            for i, source in enumerate(response['sources']):
                print(f"Source {i+1}: {source['file_name']} (Score: {source['score']:.4f})")
                # print(f"Content: {source['content'][:200]}...") # Uncomment for more detail
        else:
            print("No sources found.")

if __name__ == "__main__":
    main()
