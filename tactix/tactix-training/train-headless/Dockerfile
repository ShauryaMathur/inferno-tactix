FROM mcr.microsoft.com/playwright/python:v1.35.0-focal

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY headless_client.py .

# Create screenshots directory
RUN mkdir -p /app/screenshots

# Start the headless browser client
CMD ["python", "headless_client.py"]