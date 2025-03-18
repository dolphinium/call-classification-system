FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Update package lists and install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Default command (you can override in docker-compose.yml)
CMD ["uvicorn", "test_server:app", "--host", "0.0.0.0", "--port", "8000"]
