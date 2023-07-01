# Murli Chat App

Provides a simple Murli chat app.

# Installation

Please create a Conda environment with Python 10 and then install the following libraries with these commands:

```
pip install langchain
pip install python-dotenv
pip install streamlit
pip install openai
pip install chromadb
pip install tiktoken
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