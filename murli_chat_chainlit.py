import chainlit as cl

from dotenv import load_dotenv

from log_factory import logger
from chat_factory import init_vector_search
from question_service import extract_sources, search_and_combine

load_dotenv()

SESSION_DOCSEARCH = "SESSION_DOCSEARCH"
SOURCE_LIMIT = 512


@cl.on_chat_start
async def main():
    logger.info("Chat started")
    docsearch = init_vector_search()
    cl.user_session.set(SESSION_DOCSEARCH, docsearch)
    await cl.Message(content="Murli chat is up and running!").send()


@cl.on_message
async def main(message: str):
    docsearch = cl.user_session.get(SESSION_DOCSEARCH)
    msg_wait = cl.Message(content="")
    await msg_wait.send()

    response, similar_texts, similar_metadata = search_and_combine(
        docsearch=docsearch,
        user_question=message,
        chain_type="stuff",
        context_size=4,
    )

    # Create the streaming effect.
    msg = cl.Message(content="")
    chunk_size = 3
    for token in [
        response[i : i + chunk_size] for i in range(0, len(response), chunk_size)
    ]:
        await msg.stream_token(token)
    await msg.send()

    # Source message
    sources = extract_sources(similar_metadata)
    text_elements = []
    elements = []
    logger.info(f"similar_texts: {len(similar_texts)} similar sources: {len(sources)}")
    for text, source in zip(similar_texts, sources):
        elements.append(
            cl.Text(name=source, content=text[:SOURCE_LIMIT], display="inline")
        )

    await cl.Message(content="Sources:", elements=elements).send()


if __name__ == "__main__":
    pass