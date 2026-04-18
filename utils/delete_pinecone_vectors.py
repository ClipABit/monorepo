#!/usr/bin/env python3
"""Delete vectors from Pinecone by file_filename metadata filter."""

import argparse
import os
import sys

from pinecone import Pinecone


def main():
    parser = argparse.ArgumentParser(description="Delete Pinecone vectors by file_filename metadata")
    parser.add_argument("--index", required=True, help="Pinecone index name (e.g. prod-chunks)")
    parser.add_argument("--namespace", required=True, help="Namespace within the index")
    parser.add_argument("--filename", required=True, help="file_filename metadata value to match")
    parser.add_argument("--dry-run", action="store_true", help="List matching vectors without deleting")
    args = parser.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index)

    metadata_filter = {"file_filename": {"$eq": args.filename}}

    # Query to see what would be deleted
    results = index.query(
        vector=[0.0] * 512,
        filter=metadata_filter,
        namespace=args.namespace,
        top_k=10000,
        include_metadata=True,
    )
    print(f"Found {len(results['matches'])} vectors matching file_filename='{args.filename}'")

    matches = results["matches"]
    if len(matches) == 0:
        print("No matches found")
        return
    for match in matches:
        print(f"  id={match['id']}  score={match['score']}")
    if not args.dry_run:
        print(f"Deleting {len(matches)} vectors where file_filename='{args.filename}' from {args.index}/{args.namespace}...")
        index.delete(filter=metadata_filter, namespace=args.namespace)
        print("Done.")


if __name__ == "__main__":
    main()
