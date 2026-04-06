## FireCastBot

Python chatbot logic and optional standalone Streamlit harness for FireCastBot.

### Optional standalone harness

From the repository root:

```bash
pip install -r apps/requirements.txt
PYTHONPATH=apps streamlit run apps/app.py --server.port 8501
```

The main web app no longer depends on this Streamlit server. It now uses FireCastBot through the Flask API.

### Environment

Required:

- `GROQ_API_KEY`

Optional:

- `OPENAI_API_KEY`
- `SPEECH_TO_TEXT_PROVIDER`
- `TEXT_TO_SPEECH_PROVIDER`

`.env` and `.env.local` are loaded from the repo root, `apps/`, and `apps/chatwithme/`.
