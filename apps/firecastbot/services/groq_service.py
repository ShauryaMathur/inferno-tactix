from firecastbot.config import Settings
from firecastbot.services.llm_service import LLMService


class GroqService(LLMService):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            provider="groq",
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )
