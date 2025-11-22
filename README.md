# Hale_LLM - Dementia Care Chatbot

An elder-friendly conversational AI system built on LLaMA 3.1 8B for dementia care support.

## 🎯 Project Overview

This project provides:
1. **Chatbot API Service** - REST API for dementia care conversations
2. **Safety Guardrails** - Filters harmful, confusing, or inappropriate responses
3. **Elderly-Friendly Personalization** - Warm tone, simple language, emotional support
4. **Memory Management** - Conversation context and history tracking
5. **Fine-tuning Pipeline** - LoRA/QLoRA fine-tuning on dementia dialogue data

## 📁 Project Structure

```
Hale_LLM/
├── src/
│   ├── chatbot_api.py           # FastAPI chatbot service
│   ├── safety_guardrails.py     # Safety & personalization
│   ├── llama_inference.py       # Llama inference wrapper
│   ├── fine_tune.py             # LoRA/QLoRA fine-tuning
│   └── llama3/                  # LLaMA 3 implementation
├── data/
│   └── dementia_dialogues.jsonl # Training dataset
├── start_api.py                 # Start the API server
├── test_api.py                  # Test API endpoints
├── demo_chatbot.py              # Demo Phase 3 features
├── HOW_TO_RUN.md               # Step-by-step guide
├── README.md                    # Full documentation
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Chatbot API

Model is located at: `C:\Users\ASUS\.llama\checkpoints\Llama3.1-8B-Instruct`

```bash
python start_api.py
```

The API will be available at:
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

### 3. See Phase 3 Features in Action

Run the demo to see safety guardrails and elderly-friendly features:
```bash
python demo_chatbot.py
```

### 4. Test the API

In a new terminal:
```bash
python test_api.py
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-22T12:00:00"
}
```

### Generate Response
```http
POST /generate
```

**Request:**
```json
{
  "user_id": "user_123",
  "message": "Hello, how are you?",
  "max_tokens": 200,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "user_id": "user_123",
  "response": "Hello! I'm doing well, thank you for asking...",
  "conversation_id": "uuid-here",
  "timestamp": "2025-11-22T12:00:00"
}
```

### Get Conversation History
```http
GET /conversation/{user_id}
```

**Response:**
```json
{
  "user_id": "user_123",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-11-22T12:00:00"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you?",
      "timestamp": "2025-11-22T12:00:01"
    }
  ]
}
```

### Clear Conversation
```http
DELETE /conversation/{user_id}
```

### Update Context
```http
POST /context/update
```

## 🔗 Integration with Other Services

### Python Example

```python
import requests

API_URL = "http://localhost:8000"

# Generate response
response = requests.post(f"{API_URL}/generate", json={
    "user_id": "patient_001",
    "message": "I forgot to take my medicine",
    "max_tokens": 150,
    "temperature": 0.7
})

result = response.json()
print(f"Bot: {result['response']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

async function chat(userId, message) {
  const response = await axios.post(`${API_URL}/generate`, {
    user_id: userId,
    message: message,
    max_tokens: 150,
    temperature: 0.7
  });

  return response.data.response;
}

// Usage
chat('patient_001', 'Hello!').then(response => {
  console.log('Bot:', response);
});
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "patient_001",
    "message": "Can you help me?",
    "max_tokens": 150
  }'
```

## 🛡️ Phase 3: Safety & Personalization Features

### Safety Guardrails

The chatbot automatically filters:

**❌ Medical Advice** - Prevents diagnosing or suggesting medication changes
```
Input: "You should increase your medication dosage"
Output: "I understand you're concerned about your health. It's always best
         to talk to your doctor..."
```

**❌ Harmful Language** - Detects and softens negative phrases
```
Input: "Don't you remember? You forgot again!"
Output: "Let me remind you..."
```

**❌ Complex Language** - Simplifies technical terms
```
Input: "The pharmaceutical intervention requires implementation"
Output: "The medicine needs to be used"
```

**❌ Long Sentences** - Breaks down into short, clear sentences
```
Input: "I understand that you're feeling confused and worried..."
Output: "I understand you're feeling confused. And worried. That's okay."
```

### Elderly-Friendly Tone

**Warm Greetings**
```
Input: "Hello"
Output: "Hello, my dear! How can I help you today?"
```

**Emotional Support**
```
User: "I'm feeling sad"
Bot: "I hear you, and I'm here for you. [response]"
```

**Gentle Questions**
```
Before: "Can you tell me?"
After: "Would you like to tell me?"
```

### Configuration

Edit `src/chatbot_api.py` and `src/safety_guardrails.py` to customize:

```python
# In safety_guardrails.py
MAX_SENTENCE_LENGTH = 15  # words per sentence
MAX_RESPONSE_SENTENCES = 3  # sentences per response

# In chatbot_api.py
SYSTEM_PROMPT = """You are a caring assistant..."""
MAX_MEMORY_TURNS = 10  # conversation history
```

## 🔧 Fine-tuning (Optional)

To fine-tune the model on your dementia dialogue data:

### Prepare Data

Format your data in `data/dementia_dialogues.jsonl`:

```json
{"messages": [{"role": "user", "content": "I'm feeling confused"}, {"role": "assistant", "content": "That's okay, I'm here to help you..."}]}
```

### Run Fine-tuning

```bash
# QLoRA (recommended for 8GB+ VRAM)
python src/fine_tune.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path data/dementia_dialogues.jsonl \
    --output_dir outputs/qlora_adapter \
    --use_qlora \
    --num_epochs 3 \
    --batch_size 4
```

### Use Fine-tuned Model

Update `src/chatbot_api.py` to load the LoRA adapter.

## 🌐 Deployment

### Local Deployment

```bash
python start_api.py
```

### Production Deployment

1. **Using Gunicorn (Linux/Mac)**:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.chatbot_api:app
```

2. **Using Docker**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "start_api.py"]
```

3. **Cloud Deployment**:
- AWS EC2 with GPU
- Google Cloud Compute Engine
- Azure VM with GPU

### Environment Variables

```bash
export MODEL_PATH="/path/to/model"
export API_PORT=8000
export MAX_MEMORY_TURNS=10
```

## ⚙️ Configuration

Edit `src/chatbot_api.py` to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_MEMORY_TURNS` | 10 | Conversation history limit |
| `MAX_NEW_TOKENS` | 256 | Max response length |
| `TEMPERATURE` | 0.7 | Response creativity (0.0-1.0) |
| `MODEL_PATH` | Auto | Path to model checkpoints |

## 📊 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| RAM | 16GB | 32GB |
| Storage | 30GB | 50GB |
| CPU | 4 cores | 8+ cores |

## 🧪 Testing

Run the test suite:
```bash
python test_api.py
```

This tests:
- ✓ Health check
- ✓ Single message generation
- ✓ Multi-turn conversations
- ✓ Conversation history
- ✓ Memory management

## 🔐 Security Considerations

- **Authentication**: Add API keys or OAuth in production
- **Rate Limiting**: Implement rate limits to prevent abuse
- **Input Validation**: Already included via Pydantic models
- **HTTPS**: Use reverse proxy (nginx) with SSL in production

## 📈 Monitoring

Add logging and monitoring:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In your endpoints
logger.info(f"Request from user: {user_id}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

- **Code**: Apache 2.0
- **LLaMA Model**: Subject to Meta's LLaMA Community License Agreement

## 🙏 Acknowledgments

- Meta AI for LLaMA 3.1
- HuggingFace for transformers library
- FastAPI framework

## 📞 Support

For issues or questions:
1. Check the API docs at `/docs`
2. Review the test scripts
3. Check model checkpoint paths
4. Ensure GPU drivers are updated

---

**Status**:
- ✅ Phase 1 Complete (Model Download)
- ✅ Phase 2 Complete (API Service)
- ✅ Phase 3 Complete (Safety & Personalization)
