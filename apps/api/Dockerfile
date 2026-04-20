FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY apps/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY apps/api ./apps/api
COPY data ./data

ENV PYTHONPATH=/app/apps/api/src
ENV API_HOST=0.0.0.0
ENV API_PORT=6969

EXPOSE 6969

CMD ["python", "-m", "inferno_api"]
