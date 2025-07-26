FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python installer)
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml .
COPY uv.lock .
RUN uv pip install -r pyproject.toml --system

# Copy application code
COPY . .

# Create directories for data and vector store
RUN mkdir -p /app/data /app/chroma_langchain_db

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]