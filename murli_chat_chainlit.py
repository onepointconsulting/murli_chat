import chainlit as cl

from dotenv import load_dotenv

from log_factory import logger
from chat_factory import init_vector_search
from question_service import extract_sources, search_and_combine

load_dotenv()

SESSION_DOCSEARCH = "SESSION_DOCSEARCH"

def process_index_file():
    from chainlit import server
    from pathlib import Path

    build_path = Path(server.build_dir)
    logger.info("build_path: %s", build_path)
    if build_path.exists():
        for p in build_path.glob("*.html"):
            if p.name == "index.html":
                with open(p) as f:
                    index_contexts = f.read()
                    index_contexts = index_contexts.replace("\"/assets/", "\"assets/")
                    logger.info(index_contexts)
                with open(p, 'w') as f:
                    f.write(index_contexts)



@cl.on_chat_start
async def main():
    logger.info("Chat started")
    
    docsearch = init_vector_search()
    cl.user_session.set(SESSION_DOCSEARCH, docsearch)
    await cl.Message(content="Murli chat is up and running!").send()
    process_index_file()


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

    # Source message
    await display_sources(similar_texts, similar_metadata)

    # Create the streaming effect.
    msg = cl.Message(content="")
    chunk_size = 3
    for token in [
        response[i : i + chunk_size] for i in range(0, len(response), chunk_size)
    ]:
        await msg.stream_token(token)
    await msg.send()


async def display_sources(similar_texts, similar_metadata):
    sources = extract_sources(similar_metadata)
    found_sources = []
    elements = []
    logger.info(f"similar_texts: {len(similar_texts)} similar sources: {len(sources)}")
    for text, source in zip(similar_texts, sources):
        found_sources.append(source)
        elements.append(cl.Text(name=source, content=text, display="side"))

    await cl.Message(
        content=f"\nSources: {', '.join(found_sources)}", elements=elements
    ).send()


if __name__ == "__main__":
    pass
