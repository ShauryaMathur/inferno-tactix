import streamlit as st

from firecastbot.config import Settings
from firecastbot.prompts import (
    build_general_prompt,
    build_rag_prompt,
    build_summary_prompt,
)
from firecastbot.services.llm_service import LLMService
from firecastbot.services.speech_service import SpeechService


class ChatService:
    def __init__(
        self,
        settings: Settings,
        llm_service: LLMService,
        speech_service: SpeechService | None = None,
    ) -> None:
        self.settings = settings
        self.llm_service = llm_service
        self.speech_service = speech_service

    def handle_query(self) -> None:
        query = st.session_state.query_input
        if not query:
            return

        messages = self._build_messages(query)

        try:
            reply = self.llm_service.chat_completion(messages)
        except Exception as exc:
            st.error(f"Error with {self.settings.llm_provider} API: {exc}")
            reply = "An error occurred."

        st.session_state.conversation.append({"role": "user", "content": query})
        st.session_state.conversation.append({"role": "assistant", "content": reply})
        st.session_state.latest_output = reply
        st.session_state.latest_output_audio = None
        st.session_state.query_input = ""

        if (
            reply
            and reply != "An error occurred."
            and st.session_state.speak_responses
            and self.speech_service
        ):
            try:
                st.session_state.latest_output_audio = self.speech_service.synthesize(reply)
            except Exception as exc:
                if self.speech_service.text_to_speech_provider.output_mode == "audio_bytes":
                    st.warning(f"Unable to generate audio reply: {exc}")

    def summarize_conversation(self) -> str:
        messages = [
            {"role": "system", "content": "You are excellent at summarizing chats."},
            {"role": "user", "content": build_summary_prompt(st.session_state.conversation)},
        ]
        return self.llm_service.chat_completion(messages)

    def _build_messages(self, query: str) -> list[dict[str, str]]:
        if st.session_state.document_db:
            retriever = st.session_state.document_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.settings.retrieval_k},
            )
            context = retriever.invoke(query)
            prompt = build_rag_prompt(
                context=context,
                conversation=st.session_state.conversation,
                query=query,
            )
        else:
            prompt = build_general_prompt(
                conversation=st.session_state.conversation,
                query=query,
            )

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
