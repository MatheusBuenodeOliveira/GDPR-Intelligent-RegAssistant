"""GDPR PDF ingestion and chunking utilities.

If the official PDF is missing, a clear error is raised instructing the user to download it.
The file expected: CELEX_32016R0679_EN_TXT.pdf (official English version).
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
from typing import List
from .config import CONFIG

PDF_PATH = Path(r"g:\programação\GDPR-Intelligent-RegAssistant\CELEX_32016R0679_EN_TXT.pdf")

HEADER_PATTERN = re.compile(r"(?i)(chapter\s+[ivx]+|article\s+\d+|recital\s+\d+)")


def normalize_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()


def load_pdf() -> List[Document]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"GDPR PDF not found at {PDF_PATH}. Please download from the official source and place it there: "
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
        )
    loader = PyPDFLoader(str(PDF_PATH))
    return loader.load()


def chunk_documents(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " "],
    )
    chunks: List[Document] = []
    for d in raw_docs:
        content = normalize_text(d.page_content)
        docs = splitter.create_documents([content], metadatas=[d.metadata])
        chunks.extend(docs)
    return chunks


def header_split(raw_docs: List[Document]) -> List[Document]:
    header_docs: List[Document] = []
    for d in raw_docs:
        segments = re.split(HEADER_PATTERN, d.page_content)
        for i in range(1, len(segments), 2):
            header = segments[i].strip()
            body = segments[i + 1].strip() if i + 1 < len(segments) else ""
            if body:
                header_docs.append(Document(page_content=normalize_text(body), metadata={**d.metadata, "section_header": header}))
    return header_docs
