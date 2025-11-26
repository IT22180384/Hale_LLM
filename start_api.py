#!/usr/bin/env python3
"""
Start the Hale Dementia Care Chatbot API Server
"""

import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("=" * 70)
    print("Hale Dementia Care Chatbot API")
    print("=" * 70)
    print()
    print("Starting server...")
    print("API will be available at: http://localhost:6161")
    print("API docs at: http://localhost:6161/docs")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 70)
    print()

    # Start the server
    uvicorn.run(
        "chatbot_api:app",
        host="0.0.0.0",
        port=6161,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )


if __name__ == "__main__":
    main()
