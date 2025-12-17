# Dockerfile for Hale LLM API
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port
EXPOSE 6161

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:6161/health || exit 1

# Run the API
CMD ["python", "start_api.py"]
