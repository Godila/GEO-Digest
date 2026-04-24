FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python deps (fastapi, uvicorn already in base image or pip install)
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pyyaml

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

CMD ["python", "dashboard/app.py"]
