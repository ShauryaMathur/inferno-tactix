# Inferno-Tactix

Inferno-Tactix is now organized as an app-oriented monorepo.

## Structure

```text
apps/
  web/              React frontend
  api/              Flask inference + terrain generation API
  simulator/        WebSocket simulation / RL runtime
  headless-client/  Playwright automation client
data/
  generated/        Generated terrain assets
  models/           Runtime model artifacts
  runs/             Simulator runs
  screenshots/      Browser automation screenshots
  analytics/        Generated reports and metrics
research/
  inferno-data/     Data prep and model research code
  papers/
  reports/
ops/
  docker/
  scripts/
```

## Local development

Bootstrap dependencies:

```bash
./ops/scripts/bootstrap.sh
```

Run the app services in separate terminals:

```bash
cd apps/api && PYTHONPATH=src python -m inferno_api
cd apps/simulator && PYTHONPATH=src python -m inferno_sim
cd apps/web && npm start
```

The app then lives at:

- Web: `http://localhost:8080`
- API: `http://localhost:6969`
- WebSocket simulator: `ws://localhost:8765`

`FireCastBot` now runs through the main API and no longer depends on a standalone Streamlit service.

## RAG model comparison

The FireCastBot backend now separates:

- LLM provider/model selection
- embedding provider/model selection
- retrieval over doctrine and runtime incident indexes

That makes it possible to replay the same RAG prompt set across multiple models.

Example:

```bash
cd apps/api
PYTHONPATH=src:.. python -m inferno_api.rag_model_compare \
  --prompts evals/prompts.sample.json \
  --models evals/models.sample.json \
  --incident-report /absolute/path/to/incident_report.pdf
```

Outputs are written to:

- `apps/api/evals/rag_model_compare.json`
- `apps/api/evals/rag_model_compare.csv`
- `apps/api/evals/rag_model_compare.txt`

Prompt records can include `expected_contains`, `rubric`, and `reference_answer` so the runner can emit keyword correctness plus optional judge-based scoring alongside latency.

Provider env names used by FireCastBot and the comparison runner:

```bash
FIRECASTBOT_OPENAI_API_KEY=...
FIRECASTBOT_GROQ_API_KEY=...
FIRECASTBOT_GEMINI_API_KEY=...
FIRECASTBOT_ANTHROPIC_API_KEY=...
FIRECASTBOT_XAI_API_KEY=...
FIRECASTBOT_OLLAMA_BASE_URL=http://localhost:11434
```

Legacy names like `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, and `OLLAMA_BASE_URL` still work, but the `FIRECASTBOT_*` names are now the explicit, comparison-friendly ones.

For a local Ollama model, set:

```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=gemma4
export OLLAMA_BASE_URL=http://localhost:11434
```

Then start the API normally and FireCastBot will use your local Ollama model for generation.

## Docker development

```bash
docker compose up --build
```

## Notes

- The frontend uses environment-driven service URLs now instead of hardcoded `localhost` values in source files.
- Generated terrain assets are written to `data/generated/terrain` instead of the frontend source tree.
- The API no longer assumes it should always spawn the simulator; in containerized setups they are separate services.
