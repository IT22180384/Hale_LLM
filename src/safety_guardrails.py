"""
Safety Guardrails for Dementia Care Chatbot
Filters harmful, confusing, or inappropriate responses
"""

import re
from typing import Dict, List, Optional, Tuple


class SafetyGuardrails:
    """Safety system to prevent harmful or confusing responses"""

    # Medical advice patterns to block
    MEDICAL_PATTERNS = [
        r'\b(diagnose|diagnosis|medication dosage|prescribe|prescription)\b',
        r'\b(stop taking|discontinue|increase dose|decrease dose)\b',
        r'\b(medical emergency|call 911|go to hospital)\b',
        r'\b(symptoms of|you have|you might have)\b.*\b(disease|disorder|condition)\b',
    ]

    # Confusing/complex terms to avoid
    COMPLEX_TERMS = [
        'algorithm', 'methodology', 'paradigm', 'infrastructure',
        'optimization', 'configuration', 'implementation', 'integration',
        'neurological pathways', 'cognitive dysfunction', 'pharmaceutical',
        'therapeutic intervention', 'clinical manifestation'
    ]

    # Harmful patterns
    HARMFUL_PATTERNS = [
        r'\b(you\'re wrong|you forgot|you should know|don\'t you remember)\b',
        r'\b(that\'s incorrect|you\'re mistaken|that doesn\'t make sense)\b',
        r'\b(try harder|pay attention|focus better)\b',
    ]

    # Maximum sentence length (words)
    MAX_SENTENCE_LENGTH = 15

    # Maximum response length (sentences)
    MAX_RESPONSE_SENTENCES = 3

    def __init__(self):
        self.warnings = []

    def check_response(self, response: str) -> Tuple[bool, str, List[str]]:
        """
        Check if response is safe for elderly dementia care

        Returns:
            (is_safe, filtered_response, warnings)
        """
        self.warnings = []

        # Check for medical advice
        if self._contains_medical_advice(response):
            self.warnings.append("Contains medical advice")
            response = self._filter_medical_advice(response)

        # Check for harmful language
        if self._contains_harmful_language(response):
            self.warnings.append("Contains harmful/negative language")
            response = self._soften_language(response)

        # Check for complex terms
        response = self._simplify_language(response)

        # Check sentence length
        response = self._shorten_sentences(response)

        # Check response length
        response = self._limit_response_length(response)

        # Final safety check
        is_safe = len(self.warnings) == 0 or all(
            w != "Contains medical advice" for w in self.warnings
        )

        return is_safe, response, self.warnings

    def _contains_medical_advice(self, text: str) -> bool:
        """Check if text contains medical advice"""
        text_lower = text.lower()
        for pattern in self.MEDICAL_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _filter_medical_advice(self, text: str) -> str:
        """Replace medical advice with safe alternative"""
        return (
            "I understand you're concerned about your health. "
            "It's always best to talk to your doctor or healthcare provider "
            "about medical questions. They know your situation best and can help you. "
            "Is there anything else I can help you with today?"
        )

    def _contains_harmful_language(self, text: str) -> bool:
        """Check if text contains harmful/negative language"""
        text_lower = text.lower()
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _soften_language(self, text: str) -> str:
        """Replace harsh language with gentle alternatives"""
        replacements = {
            "you forgot": "let me remind you",
            "you should know": "let me help you remember",
            "don't you remember": "would you like me to remind you",
            "you're wrong": "let's think about this together",
            "that's incorrect": "let me help clarify",
            "try harder": "take your time",
            "pay attention": "let's focus together",
        }

        text_lower = text.lower()
        for harsh, gentle in replacements.items():
            if harsh in text_lower:
                self.warnings.append(f"Softened: '{harsh}' → '{gentle}'")

        return text

    def _simplify_language(self, text: str) -> str:
        """Replace complex terms with simpler alternatives"""
        simple_replacements = {
            'utilize': 'use',
            'facilitate': 'help',
            'demonstrate': 'show',
            'indicate': 'show',
            'approximately': 'about',
            'numerous': 'many',
            'sufficient': 'enough',
            'commence': 'start',
            'terminate': 'end',
            'endeavor': 'try',
        }

        for complex_term, simple_term in simple_replacements.items():
            if complex_term in text.lower():
                text = re.sub(
                    rf'\b{complex_term}\b',
                    simple_term,
                    text,
                    flags=re.IGNORECASE
                )
                self.warnings.append(f"Simplified: '{complex_term}' → '{simple_term}'")

        return text

    def _shorten_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones"""
        sentences = re.split(r'([.!?]+)', text)
        new_sentences = []

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else '.'

            if not sentence:
                continue

            word_count = len(sentence.split())

            # If sentence is too long, try to split it
            if word_count > self.MAX_SENTENCE_LENGTH:
                # Split on conjunctions
                parts = re.split(r'\b(and|but|or|so|because)\b', sentence)

                if len(parts) > 1:
                    # Reconstruct shorter sentences
                    current = []
                    for part in parts:
                        current.append(part)
                        if len(' '.join(current).split()) >= 8:
                            new_sentences.append(' '.join(current).strip() + '.')
                            current = []

                    if current:
                        new_sentences.append(' '.join(current).strip() + punct)

                    self.warnings.append("Split long sentence")
                else:
                    # Can't split easily, keep as is
                    new_sentences.append(sentence + punct)
            else:
                new_sentences.append(sentence + punct)

        return ' '.join(new_sentences)

    def _limit_response_length(self, text: str) -> str:
        """Limit response to maximum number of sentences"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > self.MAX_RESPONSE_SENTENCES:
            sentences = sentences[:self.MAX_RESPONSE_SENTENCES]
            self.warnings.append(
                f"Truncated response to {self.MAX_RESPONSE_SENTENCES} sentences"
            )

        return '. '.join(sentences) + '.'

    def add_reassurance(self, response: str) -> str:
        """Add reassuring elements to response"""
        reassuring_prefixes = [
            "I'm here to help. ",
            "Take your time. ",
            "That's okay. ",
            "Don't worry. ",
        ]

        reassuring_suffixes = [
            " I'm here if you need anything else.",
            " Feel free to ask me anything.",
            " I'm happy to help.",
        ]

        # Randomly add reassurance (for now, just add to longer responses)
        if len(response.split()) > 20:
            return reassuring_prefixes[0] + response + reassuring_suffixes[0]

        return response


class ElderlyToneEnhancer:
    """Enhance responses to be more elderly-friendly"""

    def __init__(self):
        self.warmth_phrases = [
            "my dear",
            "of course",
            "absolutely",
            "I understand",
            "that's perfectly fine",
            "take your time",
        ]

    def enhance_tone(self, response: str) -> str:
        """Make response more warm and patient"""

        # Add warmth to greetings
        if any(greeting in response.lower() for greeting in ['hello', 'hi', 'good morning', 'good afternoon']):
            response = response.replace('Hello', 'Hello, my dear')
            response = response.replace('Hi', 'Hello, dear')

        # Slow down the pace with ellipses (use sparingly)
        # response = response.replace('. ', '... ')

        # Make questions gentler
        response = re.sub(
            r'Can you\b',
            'Would you like to',
            response,
            flags=re.IGNORECASE
        )

        response = re.sub(
            r'Do you\b',
            'Would you',
            response,
            flags=re.IGNORECASE
        )

        return response

    def add_emotional_support(self, response: str, user_message: str) -> str:
        """Add emotional support based on user message sentiment"""

        # Detect emotional keywords
        sad_keywords = ['sad', 'lonely', 'miss', 'scared', 'worried', 'confused']
        happy_keywords = ['happy', 'good', 'great', 'wonderful', 'nice']

        user_lower = user_message.lower()

        # Add empathy for negative emotions
        if any(keyword in user_lower for keyword in sad_keywords):
            response = "I hear you, and I'm here for you. " + response

        # Celebrate positive emotions
        elif any(keyword in user_lower for keyword in happy_keywords):
            response = "That's wonderful to hear! " + response

        return response


if __name__ == "__main__":
    # Test the safety guardrails
    guardrails = SafetyGuardrails()
    tone_enhancer = ElderlyToneEnhancer()

    test_responses = [
        "You should increase your medication dosage to 50mg.",
        "Don't you remember what we talked about yesterday? You forgot again!",
        "The pharmaceutical intervention requires immediate implementation.",
        "Hello! I can help you with that. Let me explain the complex algorithm we use.",
        "I understand you're feeling sad today.",
    ]

    print("Testing Safety Guardrails:")
    print("=" * 70)

    for response in test_responses:
        print(f"\nOriginal: {response}")

        is_safe, filtered, warnings = guardrails.check_response(response)
        enhanced = tone_enhancer.enhance_tone(filtered)

        print(f"Safe: {is_safe}")
        print(f"Filtered: {enhanced}")
        if warnings:
            print(f"Warnings: {', '.join(warnings)}")

        print("-" * 70)
