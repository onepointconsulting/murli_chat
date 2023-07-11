
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path

from typing import List
import logging, re
from config import cfg

logger = logging.getLogger("murli-chat")

def load_txt(file_path: Path) -> List[Document]:
    """
    Use the csv loader to load the CSV content as a list of documents.
    :param file_path: A CSV file path
    :return: the document list after extracting and splitting all CSV records.
    """
    loader = TextLoader(file_path=str(file_path), encoding="utf-8")
    doc_list: List[Document] = loader.load()
    logger.info(f"Length of CSV list: {len(doc_list)}")
    for doc in doc_list:
        doc.page_content = re.sub(r"\n+", "\n", doc.page_content, re.MULTILINE)
    
    return split_docs(doc_list)


def split_docs(doc_list: List[Document]) -> List[Document]:
    """
    Splits the documents in smaller chunks from a list of documents.
    :param doc_list: A list of documents.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separator=cfg.chunk_separator
    )
    texts: List[Document] = text_splitter.split_documents(doc_list)
    logger.info(f"Length of texts: {len(texts)}")
    return texts
