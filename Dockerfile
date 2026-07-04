# Lightweight Azure Frontend Container
# No GPU needed - just HTTP server + file upload/download
# Size: ~500 MB (vs 3GB for full ML stack)

FROM python:3.11-slim-bookworm

WORKDIR /app

# Install minimal system dependencies (no ML libraries)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy frontend files
COPY frontend/ ./frontend/

# Copy inference modules (needed for imports, but not executed here)
COPY inference/ ./inference/
COPY inference_3d.py ./
COPY preprocessing.py ./
COPY resnet3d.py ./
COPY unet3d.py ./

# Copy Azure service
COPY service_azure.py ./
COPY requirements-azure.txt ./

# Install Python dependencies (lightweight only)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-azure.txt

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/ui || exit 1

# Run the service
ENV PORT=3000
EXPOSE 3000

CMD ["python", "-m", "uvicorn", "service_azure:app", "--host", "0.0.0.0", "--port", "3000"]