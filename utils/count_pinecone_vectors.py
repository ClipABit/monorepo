#!/usr/bin/env python3
"""Delete vectors from Pinecone by file_filename metadata filter."""

import argparse
import os
import sys

from pinecone import Pinecone


def main():
    parser = argparse.ArgumentParser(
        description="Delete Pinecone vectors by file_filename metadata"
    )
    parser.add_argument(
        "--index", required=True, help="Pinecone index name (e.g. prod-chunks)"
    )
    parser.add_argument("--namespace", required=True, help="Namespace within the index")
    parser.add_argument("--user", required=True, help="user_id metadata value to match")
    args = parser.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index)

    metadata_filter = {"user_id": {"$eq": args.user}}

    # Query to see what would be deleted
    results = index.query(
        vector=[0.0] * 512,
        filter=metadata_filter,
        namespace=args.namespace,
        top_k=10000,
        include_metadata=True,
    )
    print(f"Found {len(results['matches'])} vectors")


if __name__ == "__main__":
    main()
