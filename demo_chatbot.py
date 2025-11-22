#!/usr/bin/env python3
"""
Demo script for Hale Dementia Care Chatbot
Shows Phase 3 features: Safety Guardrails & Elderly-Friendly Tone
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from safety_guardrails import SafetyGuardrails, ElderlyToneEnhancer


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_safety_guardrails():
    """Demonstrate safety guardrails"""
    print_header("PHASE 3: Safety Guardrails Demo")

    guardrails = SafetyGuardrails()

    test_cases = [
        {
            "name": "❌ Medical Advice (BLOCKED)",
            "input": "You should increase your medication dosage to 50mg daily.",
        },
        {
            "name": "❌ Harmful Language (FILTERED)",
            "input": "Don't you remember? You forgot what we talked about yesterday!",
        },
        {
            "name": "❌ Complex Language (SIMPLIFIED)",
            "input": "The pharmaceutical intervention necessitates immediate implementation utilizing advanced methodologies.",
        },
        {
            "name": "❌ Too Long (SHORTENED)",
            "input": "I understand that you're feeling confused and worried about your situation and I want you to know that it's completely normal to feel this way and there are many things we can do to help you feel better and more comfortable in your daily life.",
        },
        {
            "name": "✅ Safe Response",
            "input": "Hello! How can I help you today? I'm here to support you.",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"\nOriginal Response:")
        print(f"  {test['input']}")

        is_safe, filtered, warnings = guardrails.check_response(test['input'])

        print(f"\nFiltered Response:")
        print(f"  {filtered}")

        print(f"\nSafety Status: {'✅ SAFE' if is_safe else '⚠️  WARNINGS'}")

        if warnings:
            print(f"Warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        print("-" * 70)


def demo_tone_enhancement():
    """Demonstrate elderly-friendly tone enhancement"""
    print_header("PHASE 3: Elderly-Friendly Tone Demo")

    tone_enhancer = ElderlyToneEnhancer()

    test_cases = [
        {
            "name": "Greeting Enhancement",
            "response": "Hello! I can help you with that.",
            "user_msg": "Hi there",
        },
        {
            "name": "Emotional Support (Sad)",
            "response": "I understand you're feeling this way. Let me help you.",
            "user_msg": "I'm feeling sad and lonely today",
        },
        {
            "name": "Emotional Support (Happy)",
            "response": "I'm glad to help you with that!",
            "user_msg": "I had a wonderful day today!",
        },
        {
            "name": "Question Softening",
            "response": "Can you tell me more about that? Do you remember?",
            "user_msg": "I need help",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"\nUser Message: {test['user_msg']}")
        print(f"\nOriginal Response:")
        print(f"  {test['response']}")

        enhanced = tone_enhancer.enhance_tone(test['response'])
        final = tone_enhancer.add_emotional_support(enhanced, test['user_msg'])

        print(f"\nEnhanced Response:")
        print(f"  {final}")

        print("-" * 70)


def demo_full_pipeline():
    """Demonstrate complete safety + tone pipeline"""
    print_header("PHASE 3: Complete Pipeline Demo")

    guardrails = SafetyGuardrails()
    tone_enhancer = ElderlyToneEnhancer()

    conversations = [
        {
            "user": "I forgot to take my medicine",
            "bot": "You should take your medication right now. Don't forget again.",
        },
        {
            "user": "I'm feeling confused about what day it is",
            "bot": "Don't you remember? It's Wednesday. You should know this.",
        },
        {
            "user": "Can you help me?",
            "bot": "Hello! I can help you with whatever you need assistance with today.",
        },
    ]

    for i, conv in enumerate(conversations, 1):
        print(f"\n[Conversation {i}]")
        print(f"\n👤 User: {conv['user']}")
        print(f"\n🤖 Raw Bot Response:")
        print(f"   {conv['bot']}")

        # Apply safety guardrails
        is_safe, filtered, warnings = guardrails.check_response(conv['bot'])

        # Apply tone enhancement
        enhanced = tone_enhancer.enhance_tone(filtered)
        final = tone_enhancer.add_emotional_support(enhanced, conv['user'])

        print(f"\n✨ Final Response (with Safety + Tone):")
        print(f"   {final}")

        if warnings:
            print(f"\n⚠️  Safety Actions Taken:")
            for warning in warnings:
                print(f"   - {warning}")

        print("-" * 70)


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  Hale Dementia Care Chatbot - Phase 3 Demo")
    print("  Safety Guardrails & Elderly-Friendly Personalization")
    print("=" * 70)

    print("\n📋 What Phase 3 Adds:")
    print("  ✓ Medical advice filtering")
    print("  ✓ Harmful language detection & softening")
    print("  ✓ Complex language simplification")
    print("  ✓ Sentence length limiting")
    print("  ✓ Warm, patient tone")
    print("  ✓ Emotional support")
    print("  ✓ Gentle question phrasing")

    input("\nPress ENTER to start demos...")

    # Run demos
    demo_safety_guardrails()

    input("\n\nPress ENTER for tone enhancement demo...")
    demo_tone_enhancement()

    input("\n\nPress ENTER for complete pipeline demo...")
    demo_full_pipeline()

    # Summary
    print_header("Summary")
    print("\n✅ Phase 3 Complete!")
    print("\nFeatures Demonstrated:")
    print("  1. ✓ Safety Guardrails - Prevents harmful/confusing responses")
    print("  2. ✓ Tone Enhancement - Warm, patient, elderly-friendly")
    print("  3. ✓ Emotional Support - Context-aware empathy")
    print("  4. ✓ Language Simplification - Clear, simple words")
    print("  5. ✓ Response Constraints - Short sentences, limited length")

    print("\n🚀 Next Steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Start API server: python start_api.py")
    print("  3. Test with: python test_api.py")
    print("  4. API will automatically use all Phase 3 features!")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
