# Inferno-Tactix

## Predicting and Fighting Wildfires Using Deep Learning and Reinforcement Learning

### Authors:
- **Shaurya Mathur**
- **Shreyas Bellary Manjunath**

---

## Overview

Inferno-Tactix is an AI-driven system designed to **predict** and **combat** wildfires via:  
- **Deep Learning** for early detection and risk assessment  
- **Reinforcement Learning** for optimal response strategies in a simulated environment  

It includes a full **data-preparation pipeline** under `inferno/` to generate the 75-day time-window datasets used for model training.

This repository provides three core services as **Docker images** under the `tactix/` directory. If you’d like to run on an HPC/CCR cluster with Singularity, you can convert these Docker images into `.sif` images after building them:

```bash
docker build -t react-client ./tactix/react-client
apptainer build rc.sif docker-daemon://react-client:latest

docker build -t python-backend ./tactix/python-backend
apptainer build pbe.sif docker-daemon://python-backend:latest

docker build -t headless-client ./tactix/headless-client
apptainer build hc.sif docker-daemon://headless-client:latest

## Use these `.sif` images for GPU-enabled training via Apptainer
```
---

## Prerequisites

### Local (Docker)
- Docker & Docker Compose

### HPC Environment
- Apptainer (Singularity) with `--nv` support for GPU passthrough  
- CUDA drivers installed on the host node  
- Write permissions to chosen host directories for logs, models, screenshots, etc.

---

## Repository Structure

```bash
Inferno-Tactix/
│
├── inferno/
│   ├── config.py                       # configuration for building the dataset
│   ├── extract_75day_data.py           # API for getting 75 day geo data window for predictions
│   ├── make_coords.py                  # file to get the coord & datetime pairs of Wildfire/No-Wildire
│   └── make_dataset.py                 # creates our custom dataset (10M+ datapoints)
│
├── tactix/                             # React app code + Dockerfile     
│   ├── tactix-training/                # Python WS server & RL trainer + Dockerfile
│   │       └── headless-client/        # Playwright client + Dockerfile
│   └── docker-compose.yaml             # brings up all three services locally
└── README.md                           # (this file)
```

## Running Locally via Docker Compose

```bash
cd tactix
docker-compose up --build
```

This brings up all three services (React, backend, headless client) and wires them automatically.

## Running on HPC/CCR with Apptainer

Use the **exact** commands below on CCR — do **not** modify them:

1. Start the React client

    ```bash
    apptainer exec \
      --pwd /app \
      rc.sif \
      env HOST=0.0.0.0 npm start
    ```
    *Binds to `0.0.0.0` so other services can connect to port `3000`.*

2. Start the Python backend (RL trainer)

    ```bash
    apptainer exec \
      --nv \
      -B ~/my_runs:/app/runs \
      -B "$(pwd)"/models:/app/models \
      --pwd /app \
      --env MODEL_DIR=/app/models \
      pbe.sif python train.py
    ```
    - `--nv`: enable GPU passthrough  
    - `~/my_runs`: writable directory for logs & monitor outputs  
    - `models/`: checkpoint directory (controlled by `MODEL_DIR`)

3. Start the headless Playwright client

    ```bash
    apptainer exec \
      -B ~/python-backend.hosts:/etc/hosts:ro \
      -B /scratch:/scratch \
      -B "$(pwd)"/screenshots:/app/screenshots \
      --pwd /app \
      hc.sif python headless_client.py
    ```
    - Custom `/etc/hosts` ensures `python-backend:8765` → `localhost`
    - python-backend.hosts is a file containing the mapping:
      ```bash
      127.0.0.1   localhost
      127.0.0.1   python-backend
      ```
    - Screenshots saved under `screenshots/`
