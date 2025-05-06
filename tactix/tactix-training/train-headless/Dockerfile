# ─────────────────────────────────────────────────────────────
# Base image: Playwright Python runtime WITH all OS deps
# (≈1 GB once Chromium is added – you said size is fine)
# ─────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/playwright/python:v1.35.0-focal

WORKDIR /app

# 1️⃣  Copy and install your Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2️⃣  Copy the application code
COPY headless_client.py .

# 3️⃣  Pre-download Chromium (and its dependencies) into /ms-playwright
#     --with-deps ensures all system libs are present too
RUN playwright install chromium --with-deps

# Optional: if you also need Firefox or WebKit, just add them:
# RUN playwright install chromium firefox webkit --with-deps

# 4️⃣  (Already present, but ensure screenshots dir exists)
RUN mkdir -p /app/screenshots

# 5️⃣  Default command
CMD ["python", "headless_client.py"]
