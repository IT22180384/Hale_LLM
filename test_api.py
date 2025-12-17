#!/usr/bin/env python3
"""
Quick test script for Hale LLM API
Run this after starting the API server
"""

import requests
import json
import time

API_URL = "http://localhost:6161"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_generate(message):
    """Test generate endpoint"""
    print(f"🤖 Testing generate endpoint...")
    print(f"📤 Input: {message}\n")

    response = requests.post(
        f"{API_URL}/generate",
        json={
            "user_id": "test-user-123",
            "message": message,
            "max_tokens": 200,
            "temperature": 0.7
        }
    )

    if response.status_code == 200:
        result = response.json()
        print("="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"\n📥 Response:\n{result['response']}\n")
        print("="*70)
        print(f"User ID: {result['user_id']}")
        print(f"Timestamp: {result['timestamp']}")
        if result.get('safety_warnings'):
            print(f"⚠️  Warnings: {result['safety_warnings']}")
        print("="*70)
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    print("="*70)
    print("Hale LLM API - Test Script")
    print("="*70)
    print()

    # Wait for server
    print("⏳ Waiting for API server to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!\n")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ Server not responding. Make sure you ran: python start_api.py")
        return

    # Run tests
    if not test_health():
        print("❌ Health check failed!")
        return

    # Test different scenarios
    test_cases = [
        "I can't remember where I put my glasses",
        "I feel confused and scared",
        "What day is it today?",
        "Hello, how are you?",
    ]

    for i, message in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'='*70}\n")
        test_generate(message)
        print()
        time.sleep(2)  # Small delay between tests

    print("\n" + "="*70)
    print("🎉 All tests completed!")
    print("="*70)

if __name__ == "__main__":
    main()
