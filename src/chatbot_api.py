"""
FastAPI-based Chatbot Service for Dementia Care
Provides conversation endpoints with memory management and elder-friendly responses.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import uuid
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from safety_guardrails import SafetyGuardrails, ElderlyToneEnhancer

app = FastAPI(
    title="Hale Dementia Care Chatbot",
    description="Elder-friendly conversational AI for dementia care support",
    version="1.0.0"
)

# Configuration
MODEL_PATH = Path.home() / ".llama" / "checkpoints" / "Llama3.1-8B-Instruct"
MAX_MEMORY_TURNS = 10  # Keep last 10 conversation turns
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

# System prompt for dementia care
SYSTEM_PROMPT = """You are a caring and patient assistant designed to support elderly individuals, especially those with dementia or memory concerns.

Your communication style:
- Use simple, clear language
- Speak slowly and warmly
- Be patient and empathetic
- Avoid complex explanations
- Provide reassurance and comfort
- Use short sentences
- Repeat information if needed
- Never rush or overwhelm

Your purpose:
- Provide companionship
- Help with daily reminders
- Offer emotional support
- Engage in gentle conversation
- Assist with memory aids

Always maintain a calm, supportive, and friendly tone."""

# Global variables (in production, use Redis or database)
model = None
tokenizer = None
conversation_memory: Dict[str, List[Dict]] = {}
safety_guard = SafetyGuardrails()
tone_enhancer = ElderlyToneEnhancer()


class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")


class GenerateRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., description="User's message")
    max_tokens: Optional[int] = Field(default=MAX_NEW_TOKENS, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=TEMPERATURE, description="Temperature for generation")


class GenerateResponse(BaseModel):
    user_id: str
    response: str
    conversation_id: str
    timestamp: str
    safety_warnings: Optional[List[str]] = Field(default=None, description="Safety warnings if any")


class ConversationHistory(BaseModel):
    user_id: str
    messages: List[Message]


def load_model():
    """Load the Llama model and tokenizer"""
    global model, tokenizer

    print(f"Loading model from {MODEL_PATH}...")

    # Check if it's Meta's official format (.pth files)
    if (MODEL_PATH / "consolidated.00.pth").exists():
        print("✓ Detected Meta's official .pth format")
        try:
            from simple_meta_loader import SimpleMetaLlama
            print("Loading with Meta's official implementation...")
            
            model = SimpleMetaLlama(
                ckpt_dir=str(MODEL_PATH),
                max_seq_len=2048,
                max_batch_size=1
            )
            tokenizer = None  # Built into SimpleMetaLlama
            return
            
        except Exception as e:
            print(f"Error loading Meta format: {e}")
            print("\nFalling back to HuggingFace...")
            import traceback
            traceback.print_exc()

    # Fallback to HuggingFace format (requires login for gated models)
    print("Loading model with HuggingFace Transformers...")
    print("Note: Llama models require HuggingFace authentication")
    
    try:
        # Try to load from local HF cache or online
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=True
        )

        print("Loading model weights... (this may take a few minutes)")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\n" + "="*70)
        print("SETUP REQUIRED:")
        print("="*70)
        print("\n1. Accept the license:")
        print("   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("\n2. Get your HuggingFace token:")
        print("   https://huggingface.co/settings/tokens")
        print("\n3. Login using:")
        print("   huggingface-cli login")
        print("\n4. Restart the server")
        print("="*70)
        raise


def get_conversation_context(user_id: str) -> List[Dict]:
    """Get conversation history for a user"""
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    # Return last N turns
    return conversation_memory[user_id][-MAX_MEMORY_TURNS:]


def add_to_memory(user_id: str, role: str, content: str):
    """Add message to conversation memory"""
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    conversation_memory[user_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })


def format_chat_prompt(messages: List[Dict]) -> str:
    """Format messages into Llama 3.1 chat format"""
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in messages:
        chat.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_response(messages: List[Dict], max_tokens: int, temperature: float) -> str:
    """Generate response using the model"""

    # Check if using SimpleMetaLlama (Meta format)
    if hasattr(model, 'generate') and tokenizer is None:
        return model.generate(messages, max_gen_len=max_tokens, temperature=temperature)

    # HuggingFace transformers
    # Format messages into chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated part (skip input tokens)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate a response to user's message

    This endpoint:
    - Maintains conversation context
    - Uses dementia-care optimized prompts
    - Returns elder-friendly responses
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get conversation history
        history = get_conversation_context(request.user_id)

        # Add user message to memory
        add_to_memory(request.user_id, "user", request.message)
        history.append({"role": "user", "content": request.message})

        # Prepare messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        # Generate response
        response_text = generate_response(
            messages,
            request.max_tokens,
            request.temperature
        )

        # Apply safety guardrails
        is_safe, filtered_response, warnings = safety_guard.check_response(response_text)

        # Enhance tone for elderly users
        enhanced_response = tone_enhancer.enhance_tone(filtered_response)

        # Add emotional support based on user message
        final_response = tone_enhancer.add_emotional_support(
            enhanced_response,
            request.message
        )

        # Add assistant response to memory
        add_to_memory(request.user_id, "assistant", final_response)

        return GenerateResponse(
            user_id=request.user_id,
            response=final_response,
            conversation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            safety_warnings=warnings if warnings else None
        )

    except Exception as e:
        print(f"ERROR in generate endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{user_id}", response_model=ConversationHistory)
async def get_conversation(user_id: str):
    """Get conversation history for a user"""
    history = get_conversation_context(user_id)

    messages = [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=msg.get("timestamp")
        )
        for msg in history
    ]

    return ConversationHistory(user_id=user_id, messages=messages)


@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Clear conversation history for a user"""
    if user_id in conversation_memory:
        del conversation_memory[user_id]
        return {"status": "success", "message": f"Conversation cleared for user {user_id}"}
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.post("/context/update")
async def update_context(user_id: str, messages: List[Message]):
    """
    Update conversation context manually
    Useful for initializing conversation with specific context
    """
    conversation_memory[user_id] = [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp or datetime.now().isoformat()
        }
        for msg in messages
    ]

    return {"status": "success", "message": f"Context updated for user {user_id}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
