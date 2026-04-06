from groq import Groq

from chatwithme.config import Settings


class GroqService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        full_response = ""
        stream = self.client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            max_tokens=self.settings.groq_max_tokens,
            temperature=self.settings.groq_temperature,
            stream=True,
        )

        for chunk in stream:
            part = chunk.choices[0].delta.content
            if part:
                full_response += part

        return full_response
