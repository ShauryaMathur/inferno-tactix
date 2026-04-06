def build_rag_prompt(*, context: object, conversation: list[dict[str, str]], query: str) -> str:
    return f"""
    Answer the user's question using the provided context. If the context lacks information, state it as general info.

    Context: {context}
    Chat History: {conversation}
    Latest Question: {query}
    """


def build_general_prompt(*, conversation: list[dict[str, str]], query: str) -> str:
    return f"""
    Answer based on the latest input. Ignore previous chat unless directly relevant.

    Chat History: {conversation}
    Latest Question: {query}
    """


def build_summary_prompt(conversation: list[dict[str, str]]) -> str:
    history_text = " ".join(
        f"{entry['role']}: {entry['content']}" for entry in conversation
    )
    return f"Summarize the following chat history:\n\n{history_text}"
