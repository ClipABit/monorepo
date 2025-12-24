#!/usr/bin/env python3
"""
Script to check the current state of the database after deletions.
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from database.pinecone_connector import PineconeConnector
from database.r2_connector import R2Connector

# Add current directory to path
sys.path.append('.')

# Load environment variables from .env file
load_dotenv()

def check_database_status():
    """Check the current state of both R2 and Pinecone databases."""
    
    # Get environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
    
    print("🔍 Environment variables check:")
    print(f"   PINECONE_API_KEY: {'✅ Set' if PINECONE_API_KEY else '❌ Missing'} (length: {len(PINECONE_API_KEY) if PINECONE_API_KEY else 0})")
    print(f"   R2_ACCOUNT_ID: {'✅ Set' if R2_ACCOUNT_ID else '❌ Missing'}")
    print(f"   ENVIRONMENT: {ENVIRONMENT}")
    
    if not all([PINECONE_API_KEY, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        print("❌ Missing required environment variables")
        return
    
    print(f"🔍 Checking database status for environment: {ENVIRONMENT}")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Initialize connectors
    pinecone_index = f"{ENVIRONMENT}-chunks"
    
    try:
        pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        print(f"✅ Connected to Pinecone index: {pinecone_index}")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return
    
    try:
        r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=ENVIRONMENT
        )
        print("✅ Connected to R2 storage")
    except Exception as e:
        print(f"❌ Failed to connect to R2: {e}")
        return
    
    print("\n" + "=" * 60)
    
    # Check Pinecone stats
    try:
        # Test basic connectivity by trying to get the index
        pinecone_connector.client.Index(pinecone_index)
        print("📊 PINECONE INDEX STATS:")
        print(f"   Index name: {pinecone_index}")
        print("   ✅ Successfully connected to index")
        
        # Try to get some sample data using the same method as the working code
        try:
            # Use the same query method that works in the main application
            sample_query = pinecone_connector.query_chunks(
                query_embedding=[0.0] * 512,  # Dummy embedding
                namespace="web-demo",
                top_k=1
            )
            print("   ✅ Query method works")
            if sample_query:
                print(f"   📊 Found {len(sample_query)} vectors in web-demo namespace")
            else:
                print("   📊 No vectors found in web-demo namespace")
        except Exception as query_e:
            print(f"   ⚠️  Query test failed: {query_e}")
            
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
    
    print("\n" + "-" * 40)
    
    # Check R2 videos
    try:
        namespace = "web-demo"
        videos = r2_connector.fetch_all_video_data(namespace)
        print(f"📹 R2 STORAGE (namespace: {namespace}):")
        print(f"   Total videos: {len(videos)}")
        
        if videos:
            print("   Videos:")
            for i, video in enumerate(videos[:10]):  # Show first 10
                filename = video.get('file_name', 'Unknown')
                identifier = video.get('hashed_identifier', 'No ID')[:20] + "..."
                print(f"     {i+1}. {filename} ({identifier})")
            
            if len(videos) > 10:
                print(f"     ... and {len(videos) - 10} more videos")
        else:
            print("   No videos found")
            
    except Exception as e:
        print(f"❌ Error checking R2 storage: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Database status check complete")

if __name__ == "__main__":
    check_database_status()