import tempfile
import time
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from firecastbot.config import Settings


class DocumentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_from_url(self, url: str):
        loader = WebBaseLoader(url)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.web_chunk_size,
            chunk_overlap=self.settings.web_chunk_overlap,
        )
        return splitter.split_documents(documents)

    def load_from_pdf(self, uploaded_file):
        start_time = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        try:
            loader = PyPDFLoader(str(temp_path))
            documents = loader.load()
        finally:
            temp_path.unlink(missing_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.pdf_chunk_size,
            chunk_overlap=self.settings.pdf_chunk_overlap,
        )
        return splitter.split_documents(documents), time.time() - start_time

    def build_vector_db(self, documents):
        start_time = time.time()
        embedder = self._build_langchain_embedder()
        db = FAISS.from_documents(documents, embedder)
        return db, time.time() - start_time

    def _build_langchain_embedder(self):
        provider = self.settings.embedding_provider.strip().casefold()
        if provider == "sentence-transformers":
            return HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"trust_remote_code": True},
            )
        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Install langchain-openai to enable OpenAI embeddings."
                ) from exc
            return OpenAIEmbeddings(
                api_key=self.settings.require_api_key("openai"),
                model=self.settings.embedding_model,
            )
        raise ValueError(f"Unsupported embedding provider: {provider}")
