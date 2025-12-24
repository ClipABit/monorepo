#!/usr/bin/env python3
"""
Test script to directly call the deletion API endpoint.
"""

import requests
import json

def test_deletion_api():
    """Test the deletion API endpoint directly."""
    
    # Use the local Modal serve endpoint
    DELETE_API_URL = "https://clipabit01--dev-server-delete-video-dev.modal.run"
    
    # Test with a real video identifier (but we'll expect 404 since we're testing)
    # Using one of the identifiers we saw in the database check
    test_identifier = "ZGV2L3dlYi1kZW1vLzE3NjYzNTgxMzFfaW52aW5jaWJsZV8zdHJhaWxlci5tb3Y="  # Last video from list
    namespace = "web-demo"
    
    print(f"🧪 Testing deletion API: {DELETE_API_URL}")
    print(f"📝 Test identifier: {test_identifier}")
    print(f"📂 Namespace: {namespace}")
    print("=" * 60)
    
    try:
        response = requests.delete(
            DELETE_API_URL,
            params={
                "hashed_identifier": test_identifier,
                "namespace": namespace
            },
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        try:
            response_data = response.json()
            print("📋 Response Body:")
            print(json.dumps(response_data, indent=2))
        except:
            print(f"📋 Response Text: {response.text}")
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ API test complete")

if __name__ == "__main__":
    test_deletion_api()