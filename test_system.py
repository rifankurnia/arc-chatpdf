#!/usr/bin/env python3
"""
Test script for the Chat With PDF system
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Test queries
TEST_QUERIES = [
    {
        "name": "Ambiguous Query",
        "query": "How many examples are enough for good accuracy?",
        "expected_behavior": "Should ask for clarification"
    },
    {
        "name": "Document Query",
        "query": "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?",
        "expected_behavior": "Should search in documents"
    },
    {
        "name": "Web Search Query",
        "query": "What did OpenAI release this month?",
        "expected_behavior": "Should perform web search"
    },
    {
        "name": "Follow-up Query",
        "query": "Can you tell me more about that?",
        "expected_behavior": "Should use session context"
    }
]


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False


def test_query(query: str, session_id: str = None) -> Dict[str, Any]:
    """Test a single query"""
    payload = {
        "query": query,
        "session_id": session_id
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Query failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Query failed: {str(e)}")
        return None


def test_session_management(session_id: str):
    """Test session management endpoints"""
    print(f"\nTesting session management for session {session_id}...")
    
    # Get session info
    try:
        response = requests.get(f"{BASE_URL}/session/{session_id}")
        if response.status_code == 200:
            print("✓ Session info retrieved")
            print(f"  Info: {response.json()}")
        else:
            print(f"✗ Failed to get session info: {response.status_code}")
    except Exception as e:
        print(f"✗ Failed to get session info: {str(e)}")
    
    # Clear session
    try:
        response = requests.delete(f"{BASE_URL}/session/{session_id}")
        if response.status_code == 200:
            print("✓ Session cleared successfully")
        else:
            print(f"✗ Failed to clear session: {response.status_code}")
    except Exception as e:
        print(f"✗ Failed to clear session: {str(e)}")


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Chat With PDF System Test")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        print("\n⚠️  API server is not running. Please start it first.")
        return
    
    print("\n" + "-" * 60)
    
    session_id = None
    
    # Test each query
    for i, test_case in enumerate(TEST_QUERIES):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected_behavior']}")
        
        result = test_query(test_case['query'], session_id)
        
        if result:
            print("✓ Query successful")
            print(f"  Response: {result['response'][:200]}...")
            print(f"  Session ID: {result['session_id']}")
            
            # Save session ID for follow-up queries
            if i == 0:
                session_id = result['session_id']
        else:
            print("✗ Query failed")
        
        # Small delay between queries
        time.sleep(1)
    
    # Test session management
    if session_id:
        test_session_management(session_id)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_debug_routing():
    """Test the debug routing endpoint"""
    print("\nTesting debug routing...")
    
    debug_queries = [
        "How many examples are enough?",
        "What did OpenAI announce today?",
        "Tell me about transformer architecture in the papers"
    ]
    
    for query in debug_queries:
        try:
            response = requests.post(
                f"{BASE_URL}/debug/routing",
                json={"query": query}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"\nQuery: {query}")
                print(f"Routing: {result['routing_result']}")
        except Exception as e:
            print(f"Debug routing failed: {str(e)}")


if __name__ == "__main__":
    run_tests()
    
    # Optional: test debug endpoint
    # test_debug_routing()