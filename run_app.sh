
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate langchain_streamlit
streamlit run ./murli_chat.py --server.port 8080
