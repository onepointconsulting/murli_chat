
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate langchain_streamlit
cd /home/ubuntu/projects/murli_chat
# /home/ubuntu/miniconda3/envs/langchain_streamlit/bin/streamlit run ./murli_chat.py --server.port 8080
/home/ubuntu/miniconda3/envs/langchain_streamlit/bin/chainlit run ./murli_chat_chainlit.py --port 8080
