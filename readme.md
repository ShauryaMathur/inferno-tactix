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

## Docker development

```bash
docker compose up --build
```

## Notes

- The frontend uses environment-driven service URLs now instead of hardcoded `localhost` values in source files.
- Generated terrain assets are written to `data/generated/terrain` instead of the frontend source tree.
- The API no longer assumes it should always spawn the simulator; in containerized setups they are separate services.
