FROM python:3.8.18

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install a compatible NumPy version first
RUN pip install --no-cache-dir numpy==1.24.3

# Install dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make sure the WebSocket server listens on 0.0.0.0
ENV WEBSOCKET_HOST=0.0.0.0
ENV WEBSOCKET_PORT=8765
ENV PYTHONUNBUFFERED=1  

# Set up a simple health check script
RUN echo '#!/bin/bash\necho "OK"' > /app/health.sh
RUN chmod +x /app/health.sh

# Start the Python training script
CMD ["python", "-u", "train.py"]