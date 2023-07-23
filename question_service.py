import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from typing import List, Dict

from log_factory import logger

from config import cfg


def enhance_question(orig_question: str) -> str:
    return f"According to the context: {orig_question}"


def process_question(
    similar_docs: List[Document], user_question: str, chain_type: str = "map_reduce"
) -> str:
    """
    Sends the question to the LLM.
    :param similar_docs: A list of documents with the documents retrieved from the vector database.
    :param user_question: A user question
    :return: The result computed by the LLM.
    """
    # See https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
    chain = load_qa_chain(cfg.llm, chain_type=chain_type)
    similar_texts = [d.page_content for d in similar_docs]
    with get_openai_callback() as callback:
        response = chain.run(
            input_documents=similar_docs, question=enhance_question(user_question)
        )
        logger.info(callback)
    similar_metadata = [d.metadata for d in similar_docs]
    return response, similar_texts, similar_metadata


def search_and_combine(
    docsearch: FAISS,
    user_question: str,
    chain_type: str = "stuff",
    context_size: int = 4,
):
    similar_docs: List[Document] = docsearch.similarity_search(
        user_question, k=context_size
    )
    response, similar_texts, similar_metadata = process_question(
        similar_docs, user_question, chain_type
    )
    return response, similar_texts, similar_metadata


def extract_sources(similar_metadata: List[Dict]):
    sources = []
    for metadata in similar_metadata:
        source = metadata["source"]
        sources.append(re.sub(r".+[\\/](.+)\.txt", r"\1", source))
    return sources


if __name__ == "__main__":
    similar_metadata = [
        {
            "source": "C:\\development\\playground\\langchain\\murli_en_chat\\data\\murli_en_2002-11-23_avyakt.txt"
        },
        {
            "source": "C:\\development\\playground\\langchain\\murli_en_chat\\data\\murli_en_1972-07-16_avyakt_2011-03-20.txt"
        },
    ]
    logger.info(extract_sources(similar_metadata=similar_metadata))
