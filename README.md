# Murli Chat App

Provides a simple Murli chat app.

# Installation

On Linux systems you might need to install g++ before installing ChromaDB.

```
sudo apt install g++
```

Please create a Conda environment (e.g. langchain_streamlit) with Python 10 and then install the following libraries with these commands:

```
conda activate langchain_streamlit
pip install langchain
pip install python-dotenv
pip install streamlit
pip install openai
pip install chromadb
pip install tiktoken
pip install chainlit

```

We have used the following library versions:

```
langchain==0.0.215
python-dotenv==1.0.0
streamlit==1.23.1
openai==0.27.8
chromadb==0.3.26
tiktoken==0.4.0
```

Ensure you have a .env file with these variables;

```
OPENAI_API_KEY=<key>
DOC_LOCATION=<path to the text murlis>
```

# Running the app

This is the command which runs the app on port 8080

### streamlit:
```
streamlit run .\murli_chat.py --server.port 8080
```

### chainlit
```
chainlit run ./murli_chat_chainlit.py -w --port 8081
chainlit run ./murli_chat_chainlit.py --port 8081
```

