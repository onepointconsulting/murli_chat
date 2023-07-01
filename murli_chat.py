from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

import shutil
import streamlit as st
from dotenv import load_dotenv

import logging
import os


load_dotenv()

logging.basicConfig(level='INFO')

class Config():
    chunk_size = 4000
    chunk_overlap = 100
    chunk_separator = "\n\n"
    chroma_persist_directory = os.environ['CHROMA_STORE']
    embeddings = OpenAIEmbeddings(chunk_size=25)
    model = 'gpt-3.5-turbo-16k'
    # model = 'gpt-4'
    llm = ChatOpenAI(model=model, temperature=0)
    history_file = Path('chat_history.txt')
    # To overcome rate limiting erros, please use https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
    search_results = 5

cfg = Config()

logger = logging.getLogger("murli-chat")

def load_txt(file_path: Path) -> List[Document]:
    """
    Use the csv loader to load the CSV content as a list of documents.
    :param file_path: A CSV file path
    :return: the document list after extracting and splitting all CSV records.
    """
    loader = TextLoader(file_path=str(file_path), encoding="utf-8")
    doc_list: List[Document] = loader.load()
    logger.info(f"First item: {doc_list[0].page_content}")
    logger.info(f"Length of CSV list: {len(doc_list)}")
    return split_docs(doc_list)


def split_docs(doc_list: List[Document]) -> List[Document]:
    """
    Splits the documents in smaller chunks from a list of documents.
    :param doc_list: A list of documents.
    """
    text_splitter = CharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, separator=cfg.chunk_separator)
    texts = text_splitter.split_documents(doc_list)
    return texts


def extract_embeddings(texts: List[Document], doc_path: Path) -> Chroma:
    """
    Either saves the Chroma embeddings locally or reads them from disk, in case they exist.
    :return a Chroma wrapper around the embeddings.
    """
    embedding_dir = f"{cfg.chroma_persist_directory}/{doc_path.stem}"
    if Path(embedding_dir).exists():
        return Chroma(persist_directory=embedding_dir, embedding_function=cfg.embeddings)
    # if Path(embedding_dir).exists():
    #     shutil.rmtree(embedding_dir, ignore_errors=True)
    try:
        docsearch = Chroma.from_documents(texts, cfg.embeddings, persist_directory=embedding_dir)
        docsearch.persist()
    except Exception as e:
        logger.error(f"Failed to process {doc_path}: {str(e)}")
        if 'docsearch' in vars() or 'docsearch' in globals():
            docsearch.persist()
        return docsearch
    return docsearch


def enhance_question(orig_question: str) -> str: 
    return f"According to the context: {orig_question}"


def process_question(similar_docs: List[Document], user_question: str) -> str:
    """
    Sends the question to the LLM.
    :param similar_docs: A list of documents with the documents retrieved from the vector database.
    :param user_question: A user question
    :return: The result computed by the LLM.
    """
    # See https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
    chain = load_qa_chain(cfg.llm, chain_type='map_reduce')
    similar_texts = [d.page_content for d in similar_docs]
    with get_openai_callback() as callback:
        response = chain.run(input_documents=similar_docs, question=enhance_question(user_question))
        logger.info(callback)
    return response, similar_texts


def write_history(question):
    """
    Writes the question into a local history file.
    :param question: The text to be written to the local history file.
    """
    if len(question) > 0:
        with open(cfg.history_file, "a") as f:
            f.write(f"{question}\n")


@st.cache_data()
def read_history()-> List[str]:
    """
    Reads and caches some historical questions. Which you can use to ask questions in the UI.
    :return: a list of questions.
    """
    with open(cfg.history_file, "r") as f:
        return list(set([l for l in f.readlines() if len(l.strip()) > 0]))
    

def process_user_question(docsearch: Chroma, user_question: str):
    """
    Receives a user question and searches for similar text documents in the vector database.
    Using the similar texts and the user question retrieves the response from the LLM.
    :param docsearch: The reference to the vector database object
    :param user_question: The question the user has typed.
    """
    if user_question:
        similar_docs: List[Document] = docsearch.similarity_search(user_question, k = cfg.search_results)
        response, similar_texts = process_question(similar_docs, user_question)
        st.markdown(response)
        if len(similar_texts) > 0:
            write_history(user_question)
            st.text("Similar entries (Vector database results)")
            st.write(similar_texts)
        else:
            st.warning("This answer is unrelated to our context.")
    

def init_streamlit(docsearch: Chroma):
    """
    Creates the Streamlit user interface.
    This code expects some form of user question and as soon as it is there it processes
    the question.
    It can also process a question from a drop down with pre-defined questions.
    Use streamlit like this:
    streamlit run ./murli_chat.py
    """
    title = "Ask questions about the Murlis"
    st.set_page_config(page_title=title)
    st.header(title)
    # st.write(f"Context with {len(texts)} entries")
    simple_chat_tab, historical_tab = st.tabs(["Simple Chat", "Historical Questions"])
    with simple_chat_tab:
        user_question = st.text_input("Your question")
        with st.spinner('Please wait ...'):
            process_user_question(docsearch=docsearch, user_question=user_question)
    with historical_tab:
        user_question_2 = st.selectbox("Ask a previous question", read_history())
        with st.spinner('Please wait ...'):
            logger.info(f"question: {user_question_2}")
            process_user_question(docsearch=docsearch, user_question=user_question_2)


def load_texts(doc_location: str) -> Tuple[List[str], Path]:
    """
    Loads the texts of the CSV file and concatenates all texts in a single list.
    :param doc_location: The document location.
    :return: a tuple with a list of strings and a path.
    """
    doc_path = Path(doc_location)
    texts = []
    for i, p in enumerate(doc_path.glob("*.txt")):
        try:
            logger.info(f"Processed {i}")
            texts.extend(load_txt(p))
        except Exception as e:
            logger.error(f"Cannot process {p} due to {e}")
    logger.info(f"Length of texts: {len(texts)}")
    return texts, doc_path


def main(doc_location: str ='onepoint_chat'):
    """
    Main entry point for the application.
    It loads all texts from a specific folder and specific web pages, 
    creates the vector database and initializes the user interface.
    :param doc_location: The location of the CSV files
    """
    logger.info(f"Using doc location {doc_location}.")
    doc_path = Path(doc_location)
    embedding_dir = f"{cfg.chroma_persist_directory}/{doc_path.stem}"
    if Path(embedding_dir).exists():
        logger.info(f"reading from existing directory")
        docsearch = Chroma(persist_directory=embedding_dir, embedding_function=cfg.embeddings)
    else:
        logger.info(f"Generating vectors")
        # texts, doc_path = load_texts(doc_location=doc_location)
        # docsearch = extract_embeddings(texts=texts, doc_path=Path(doc_path))
    init_streamlit(docsearch=docsearch)


if __name__ == "__main__":
    main(os.environ['DOC_LOCATION'])