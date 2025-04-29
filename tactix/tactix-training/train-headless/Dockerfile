# Pyppeteer Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install dependencies for Pyppeteer and Chromium
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    fonts-liberation \
    libappindicator3-1 \
    libnss3 \
    lsb-release \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Pyppeteer (Python version of Puppeteer)
RUN pip install pyppeteer

COPY . .

CMD ["python", "infernoenv.py"]
