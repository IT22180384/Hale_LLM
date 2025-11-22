#!/usr/bin/env python3
"""
Test the Chatbot API
"""

import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing /health endpoint...")
    print("="*70)

    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_generate(user_id="test_user_123", message="Hello, how are you today?"):
    """Test generate endpoint"""
    print("\n" + "="*70)
    print("Testing /generate endpoint...")
    print("="*70)

    data = {
        "user_id": user_id,
        "message": message,
        "max_tokens": 200,
        "temperature": 0.7
    }

    print(f"Request: {json.dumps(data, indent=2)}")

    response = requests.post(f"{API_URL}/generate", json=data)
    print(f"\nStatus: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nUser: {message}")
        print(f"Bot: {result['response']}")
        print(f"\nConversation ID: {result['conversation_id']}")
        print(f"Timestamp: {result['timestamp']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_conversation_history(user_id="test_user_123"):
    """Test conversation history endpoint"""
    print("\n" + "="*70)
    print(f"Testing /conversation/{user_id} endpoint...")
    print("="*70)

    response = requests.get(f"{API_URL}/conversation/{user_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nConversation history for {user_id}:")
        print(json.dumps(result, indent=2))
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_conversation_flow():
    """Test a full conversation flow"""
    print("\n" + "="*70)
    print("Testing Full Conversation Flow...")
    print("="*70)

    user_id = f"test_user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Conversation 1
    print("\n[Turn 1]")
    test_generate(user_id, "Hello! What's your name?")

    # Conversation 2
    print("\n[Turn 2]")
    test_generate(user_id, "Can you help me remember to take my medicine?")

    # Conversation 3
    print("\n[Turn 3]")
    test_generate(user_id, "What time is it?")

    # Get full history
    print("\n[Full History]")
    test_conversation_history(user_id)


def main():
    print("="*70)
    print("Hale Dementia Care Chatbot API - Test Suite")
    print("="*70)
    print(f"\nAPI URL: {API_URL}")
    print("\nMake sure the API server is running!")
    print("Run: python start_api.py")
    print()

    input("Press ENTER to start tests...")

    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Single Generation", lambda: test_generate("demo_user", "Tell me about yourself")),
        ("Conversation Flow", test_conversation_flow),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))

    # Print summary
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)

    for test_name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
