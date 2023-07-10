
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate langchain_streamlit
cd /home/ubuntu/projects/murli_chat
streamlit run ./murli_chat.py --server.port 8080
