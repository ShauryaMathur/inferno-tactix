import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_dotenv_fallback(dotenv_path: Path) -> None:
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 2048
    llm_temperature: float = 1.3
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 2048
    groq_temperature: float = 1.3
    retrieval_k: int = 3
    chat_recent_turn_limit: int = 6
    chat_summarize_after_turns: int = 8
    pdf_chunk_size: int = 2000
    pdf_chunk_overlap: int = 100
    web_chunk_size: int = 200
    web_chunk_overlap: int = 20
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    speech_to_text_provider: str = "browser"
    text_to_speech_provider: str = "browser"
    speech_to_text_model: str = "gpt-4o-mini-transcribe"
    text_to_speech_model: str = "gpt-4o-mini-tts"
    text_to_speech_voice: str = "alloy"
    text_to_speech_format: str = "mp3"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    xai_base_url: str = "https://api.x.ai/v1"
    ollama_base_url: str = "http://localhost:11434"

    def require_api_key(self, provider: str) -> str:
        provider_key_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "xai": self.xai_api_key,
        }
        api_key = provider_key_map.get(provider.strip().casefold())
        if not api_key:
            raise ValueError(
                f"Missing API key for provider '{provider}'. Add it to your environment or .env file."
            )
        return api_key


def get_settings() -> Settings:
    package_root = Path(__file__).resolve().parent
    apps_root = package_root.parent
    repo_root = apps_root.parent

    for env_path in (
        repo_root / ".env",
        repo_root / ".env.local",
        apps_root / ".env",
        apps_root / ".env.local",
        package_root / ".env",
        package_root / ".env.local",
    ):
        if env_path.exists():
            if load_dotenv is not None:
                load_dotenv(dotenv_path=env_path, override=True)
            else:
                _load_dotenv_fallback(env_path)

    groq_api_key = (
        os.getenv("FIRECASTBOT_GROQ_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )
    openai_api_key = (
        os.getenv("FIRECASTBOT_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    openrouter_api_key = (
        os.getenv("FIRECASTBOT_OPENROUTER_API_KEY")
        or os.getenv("OPEN_ROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    anthropic_api_key = (
        os.getenv("FIRECASTBOT_ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )
    gemini_api_key = (
        os.getenv("FIRECASTBOT_GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    xai_api_key = (
        os.getenv("FIRECASTBOT_XAI_API_KEY")
        or os.getenv("XAI_API_KEY")
        or os.getenv("GROK_API_KEY")
    )
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    llm_model = os.getenv("LLM_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", os.getenv("GROQ_MAX_TOKENS", "2048")))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "1.3")))
    groq_model = os.getenv("GROQ_MODEL", llm_model if llm_provider.strip().casefold() == "groq" else "llama-3.3-70b-versatile")
    groq_max_tokens = int(os.getenv("GROQ_MAX_TOKENS", str(llm_max_tokens)))
    groq_temperature = float(os.getenv("GROQ_TEMPERATURE", str(llm_temperature)))
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chat_recent_turn_limit = int(os.getenv("CHAT_RECENT_TURN_LIMIT", "6"))
    chat_summarize_after_turns = int(os.getenv("CHAT_SUMMARIZE_AFTER_TURNS", "8"))
    speech_to_text_provider = os.getenv("SPEECH_TO_TEXT_PROVIDER", "browser")
    text_to_speech_provider = os.getenv("TEXT_TO_SPEECH_PROVIDER", "browser")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    xai_base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    ollama_base_url = os.getenv(
        "FIRECASTBOT_OLLAMA_BASE_URL",
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    return Settings(
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        xai_api_key=xai_api_key,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        llm_temperature=llm_temperature,
        groq_model=groq_model,
        groq_max_tokens=groq_max_tokens,
        groq_temperature=groq_temperature,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        chat_recent_turn_limit=chat_recent_turn_limit,
        chat_summarize_after_turns=chat_summarize_after_turns,
        speech_to_text_provider=speech_to_text_provider,
        text_to_speech_provider=text_to_speech_provider,
        openrouter_base_url=openrouter_base_url,
        xai_base_url=xai_base_url,
        ollama_base_url=ollama_base_url,
    )
