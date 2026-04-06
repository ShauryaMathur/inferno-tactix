import streamlit as st

from chatwithme.config import get_settings
from chatwithme.services.chat_service import ChatService
from chatwithme.services.document_service import DocumentService
from chatwithme.services.groq_service import GroqService
from chatwithme.services.speech_service import SpeechService
from chatwithme.session import initialize_session_state


def render_instructions() -> None:
    with st.expander("Instructions"):
        st.markdown(
            """
            1. Open the sidebar menu.
            2. Upload a PDF or enter a web URL.
            3. After uploading/loading, click 'Create Vector Store'.
            4. Optionally transcribe a question with OpenAI or your browser.
            5. Enter your query and submit.
            6. Optionally enable spoken responses with OpenAI or your browser.
            7. Click 'Generate Chat Summary' for a session summary.
            """
        )


def render_sidebar(document_service: DocumentService) -> None:
    st.sidebar.subheader("Choose document source")
    choice = st.sidebar.radio("Select:", ("Upload PDF", "Enter Web URL"))

    if choice == "Upload PDF":
        pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file and st.session_state.loaded_docs is None:
            with st.spinner("Loading PDF..."):
                documents, elapsed_time = document_service.load_from_pdf(pdf_file)
                st.session_state.loaded_docs = documents
            st.success(f"PDF loaded in {elapsed_time:.2f} seconds")

    if choice == "Enter Web URL":
        url = st.sidebar.text_input("Enter URL", key="url_input")
        if st.sidebar.button("Process URL") and url:
            with st.spinner("Fetching URL content..."):
                st.session_state.loaded_docs = document_service.load_from_url(url)

    if st.session_state.loaded_docs and st.sidebar.button("Create Vector Store"):
        with st.spinner("Building vector store..."):
            vector_db, elapsed_time = document_service.build_vector_db(
                st.session_state.loaded_docs
            )
            st.session_state.document_db = vector_db
        st.success(f"Vector DB created in {elapsed_time:.2f} seconds")


def render_speech_provider_settings(
    speech_service_options: dict[str, str],
    default_speech_to_text_provider_id: str,
    default_text_to_speech_provider_id: str,
) -> tuple[str, str]:
    st.sidebar.subheader("Speech providers")
    provider_ids = list(speech_service_options)

    if not st.session_state.speech_to_text_provider_id:
        st.session_state.speech_to_text_provider_id = default_speech_to_text_provider_id
    if not st.session_state.text_to_speech_provider_id:
        st.session_state.text_to_speech_provider_id = default_text_to_speech_provider_id

    selected_stt_provider_id = st.sidebar.selectbox(
        "Speech to text",
        options=provider_ids,
        format_func=lambda provider_id: speech_service_options[provider_id],
        key="speech_to_text_provider_id",
    )
    selected_tts_provider_id = st.sidebar.selectbox(
        "Text to speech",
        options=provider_ids,
        format_func=lambda provider_id: speech_service_options[provider_id],
        key="text_to_speech_provider_id",
    )

    return selected_stt_provider_id, selected_tts_provider_id


def render_audio_transcription_input(speech_service: SpeechService) -> None:
    audio_input_widget = getattr(st, "audio_input", None)
    if callable(audio_input_widget):
        recorded_audio = audio_input_widget("Record your question")
    else:
        recorded_audio = st.file_uploader(
            "Upload an audio question",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
            key="audio_query_input",
        )

    if recorded_audio is not None and st.button("Transcribe Audio"):
        if not speech_service.transcription_available:
            st.error(
                speech_service.transcription_unavailable_reason
                or "Speech transcription is unavailable."
            )
            return

        with st.spinner("Transcribing audio..."):
            try:
                transcript = speech_service.transcribe(recorded_audio)
            except Exception as exc:
                st.error(f"Unable to transcribe audio: {exc}")
            else:
                st.session_state.query_input = transcript
                st.session_state.latest_transcript = transcript
                st.success("Transcript added to the question box.")


def render_browser_speech(speech_service: SpeechService) -> None:
    browser_event = speech_service.render_browser_widget(
        speak_text=st.session_state.latest_output,
        auto_speak=st.session_state.speak_responses,
    )
    if browser_event is None:
        return

    if (
        browser_event.event_id
        and browser_event.event_id != st.session_state.last_browser_speech_event_id
    ):
        st.session_state.last_browser_speech_event_id = browser_event.event_id
        if browser_event.transcript:
            st.session_state.query_input = browser_event.transcript
            st.session_state.latest_transcript = browser_event.transcript
            st.success("Transcript added to the question box.")
        elif browser_event.error:
            st.error(browser_event.error)

    if (
        speech_service.speech_to_text_provider.input_mode == "browser"
        and not browser_event.supports_recognition
    ):
        st.caption("Your browser does not expose the Web Speech recognition API.")

    if (
        speech_service.text_to_speech_provider.output_mode == "browser"
        and st.session_state.speak_responses
        and not browser_event.supports_synthesis
    ):
        st.caption("Your browser does not expose speech synthesis.")


def render_chat(chat_service: ChatService, speech_service: SpeechService) -> None:
    st.subheader("Ask a question")

    if speech_service.synthesis_available:
        st.checkbox("Read responses aloud", key="speak_responses")
    else:
        st.checkbox("Read responses aloud", key="speak_responses", disabled=True)
        if speech_service.synthesis_unavailable_reason:
            st.caption(speech_service.synthesis_unavailable_reason)

    if (
        speech_service.speech_to_text_provider.input_mode == "browser"
        or speech_service.text_to_speech_provider.output_mode == "browser"
    ):
        render_browser_speech(speech_service)

    if speech_service.speech_to_text_provider.input_mode == "upload":
        render_audio_transcription_input(speech_service)

    if st.session_state.latest_transcript:
        st.caption(f"Latest transcript: {st.session_state.latest_transcript}")

    st.text_area("Enter your question:", key="query_input")
    st.button("Submit", on_click=chat_service.handle_query)

    if st.session_state.latest_output:
        st.write(st.session_state.latest_output)
        if st.session_state.latest_output_audio:
            st.audio(
                st.session_state.latest_output_audio,
                format=f"audio/{chat_service.settings.text_to_speech_format}",
            )

    if st.button("Generate Chat Summary"):
        with st.spinner("Summarizing chat..."):
            st.session_state.chat_summary = chat_service.summarize_conversation()

    if st.session_state.chat_summary:
        with st.expander("Chat Summary"):
            st.write(st.session_state.chat_summary)

    with st.expander("Recent Chat History"):
        for entry in reversed(st.session_state.conversation[-8:]):
            st.write(f"{entry['role'].capitalize()}: {entry['content']}")


def main() -> None:
    st.set_page_config(page_title="Chatbot")
    st.title("FireCastRL ChatBot")
    initialize_session_state()

    try:
        settings = get_settings()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    selected_stt_provider_id, selected_tts_provider_id = render_speech_provider_settings(
        SpeechService.provider_options(),
        settings.speech_to_text_provider,
        settings.text_to_speech_provider,
    )

    document_service = DocumentService(settings)
    groq_service = GroqService(settings)
    speech_service = SpeechService(
        settings,
        speech_to_text_provider_id=selected_stt_provider_id,
        text_to_speech_provider_id=selected_tts_provider_id,
    )
    chat_service = ChatService(settings, groq_service, speech_service)

    render_instructions()
    render_sidebar(document_service)
    render_chat(chat_service, speech_service)
