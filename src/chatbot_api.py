"""
FastAPI-based Chatbot Service for Dementia Care
Provides conversation endpoints with memory management and elder-friendly responses.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
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
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# ADAPTER_PATH can be either local path or HuggingFace model ID
# Set via environment variable ADAPTER_MODEL_ID or use local path
import os
ADAPTER_MODEL_ID = os.getenv("ADAPTER_MODEL_ID", None)
ADAPTER_PATH = ADAPTER_MODEL_ID if ADAPTER_MODEL_ID else (Path(__file__).parent / "models" / "phi_pitt_lora_final")
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
    """Load the fine-tuned Llama model with LoRA adapter (Mac optimized)"""
    global model, tokenizer

    print("🤖 Loading fine-tuned model with LoRA adapter...")

    try:
        # Determine device (Mac uses CPU, GPU systems use CUDA)
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
            # Use quantization only on GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("📥 Loading base model (GPU mode with 4-bit quantization)...")
        else:
            device = "cpu"
            dtype = torch.float32
            bnb_config = None
            print("📥 Loading base model (CPU mode - Mac)...")

        # Load base model
        if bnb_config:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        # Load tokenizer
        print("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load LoRA adapter
        print(f"🎯 Loading LoRA adapter from {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n" + "="*70)
        print("SETUP REQUIRED:")
        print("="*70)
        print("\n1. Make sure the LoRA adapter exists at:")
        print(f"   {ADAPTER_PATH}")
        print("\n2. Accept the LLaMA license:")
        print("   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        print("\n3. Get your HuggingFace token:")
        print("   https://huggingface.co/settings/tokens")
        print("\n4. Login using:")
        print("   huggingface-cli login")
        print("\n5. Restart the server")
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
    """Generate response using the fine-tuned model"""

    # Format messages into Llama 3 chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with fine-tuned model
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
