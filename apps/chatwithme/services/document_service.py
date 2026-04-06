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

from chatwithme.config import Settings


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
        embedder = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={"trust_remote_code": True},
        )
        db = FAISS.from_documents(documents, embedder)
        return db, time.time() - start_time
