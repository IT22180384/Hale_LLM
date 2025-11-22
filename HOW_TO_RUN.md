# 🚀 How to Run the Chatbot - Step by Step

## ✅ What You Have Now

- ✅ Llama 3.1 8B Instruct downloaded
- ✅ Complete chatbot API with safety features
- ✅ All Phase 1, 2, and 3 features ready!

## 📋 Commands to Run

### Step 1: Install Dependencies

Open PowerShell/Command Prompt and run:

```powershell
cd c:\research\Hale_LLM
pip install -r requirements.txt
```

This will install:
- FastAPI (API framework)
- Transformers (for Llama model)
- Safety guardrails dependencies
- All other required packages

**Time**: ~2-5 minutes

---

### Step 2: Verify Model Installation (Optional)

Check if the model is properly installed:

```powershell
python check_model.py
```

This will verify:
- ✓ Model directory exists
- ✓ All required Meta format files present
- ✓ MetaLlamaWrapper can be imported

**Time**: ~5 seconds

---

### Step 3: See Phase 3 Features (Demo)

Run the demo to see safety guardrails in action:

```powershell
python demo_chatbot.py
```

This shows:
- ✓ Medical advice filtering
- ✓ Harmful language detection
- ✓ Complex language simplification
- ✓ Elderly-friendly tone enhancement

**No model required** - just shows how the safety features work!

**Time**: ~1 minute

---

### Step 3: Start the Chatbot API

Start the server:

```powershell
python start_api.py
```

You'll see:
```
Hale Dementia Care Chatbot API
======================================
Starting server...
API will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

Press CTRL+C to stop the server
======================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
Loading model from C:\Users\ASUS\.llama\checkpoints\Llama3.1-8B-Instruct...
✓ Model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**⚠️ Note**: Model loading takes 1-2 minutes depending on your GPU/CPU

Leave this terminal open!

---

### Step 4: Test the API

Open a **NEW** terminal/PowerShell window:

```powershell
cd c:\research\Hale_LLM
python test_api.py
```

This will:
1. ✓ Check health endpoint
2. ✓ Send test messages
3. ✓ Test conversation flow
4. ✓ Show safety features working

**Sample Output**:
```
Testing /generate endpoint...
============================================

User: Hello, how are you today?
Bot: Hello, my dear! I'm doing well, thank you for asking...

Conversation ID: abc-123-def
```

---

### Step 5: Use the API from Your Code

#### Python Example:

```python
import requests

API_URL = "http://localhost:8000"

# Send a message
response = requests.post(f"{API_URL}/generate", json={
    "user_id": "patient_001",
    "message": "I'm feeling confused today",
    "max_tokens": 200,
    "temperature": 0.7
})

result = response.json()
print(f"Bot: {result['response']}")
print(f"Safety warnings: {result.get('safety_warnings', 'None')}")
```

#### Test with cURL:

```bash
curl -X POST "http://localhost:8000/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": \"test_user\", \"message\": \"Hello!\"}"
```

---

## 🌐 API Endpoints

Once the server is running:

### Interactive API Documentation
Open your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can **test the API directly** in your browser!

### Available Endpoints:

1. **Health Check**
   ```
   GET http://localhost:8000/health
   ```

2. **Generate Response**
   ```
   POST http://localhost:8000/generate
   Body: {
     "user_id": "user_123",
     "message": "Hello!",
     "max_tokens": 200,
     "temperature": 0.7
   }
   ```

3. **Get Conversation History**
   ```
   GET http://localhost:8000/conversation/user_123
   ```

4. **Clear Conversation**
   ```
   DELETE http://localhost:8000/conversation/user_123
   ```

---

## 🔧 Troubleshooting

### Issue: "Model not found"
**Solution**: Model should be at:
```
C:\Users\ASUS\.llama\checkpoints\Llama3.1-8B-Instruct
```
Check with:
```powershell
dir C:\Users\ASUS\.llama\checkpoints\Llama3.1-8B-Instruct
```
Should show: `consolidated.00.pth`, `params.json`, `tokenizer.model`

### Issue: "Out of memory"
**Solution**: You need at least 8GB VRAM (GPU) or 16GB RAM (CPU)
- Close other applications
- Or use a smaller model (3B version)

### Issue: "Port already in use"
**Solution**: Change the port in `start_api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change to 8001
```

### Issue: "Import error"
**Solution**: Install dependencies again:
```powershell
pip install -r requirements.txt --upgrade
```

---

## 📊 What Happens When You Run It?

### 1. Model Loading (First time - slow)
```
Loading model...
✓ Model loaded successfully
```

### 2. You Send a Message
```json
{
  "user_id": "patient_001",
  "message": "I forgot my medicine"
}
```

### 3. Processing Pipeline

**Step 1**: Get conversation history
**Step 2**: Generate response with Llama model
**Step 3**: Apply safety guardrails
- ✓ Check for medical advice
- ✓ Check for harmful language
- ✓ Simplify complex terms
- ✓ Shorten long sentences

**Step 4**: Enhance tone
- ✓ Add warmth
- ✓ Add emotional support
- ✓ Soften questions

**Step 5**: Return response
```json
{
  "response": "I hear you. Let me remind you about your medicine...",
  "safety_warnings": ["Simplified: 'medication' → 'medicine'"]
}
```

---

## 🎯 Integration with Your Dementia Prediction Project

From your other project, call this API:

```python
# In your dementia prediction project
import requests

def get_chatbot_response(user_id, message):
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "user_id": user_id,
            "message": message,
            "max_tokens": 200
        }
    )
    return response.json()["response"]

# Usage
bot_reply = get_chatbot_response("patient_123", "Hello!")
print(bot_reply)
```

---

## 📱 Quick Command Reference

| Action | Command |
|--------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| See Phase 3 demo | `python demo_chatbot.py` |
| Start API server | `python start_api.py` |
| Test API | `python test_api.py` |
| View API docs | Open http://localhost:8000/docs |
| Stop server | Press `CTRL+C` in server terminal |

---

## ✅ You're All Set!

Your chatbot is now:
- ✅ Safe (filters harmful content)
- ✅ Elderly-friendly (warm, simple language)
- ✅ Memory-aware (tracks conversations)
- ✅ API-ready (integrate anywhere)

**Everything is in this one repo - no separate repos needed!** 🎉
