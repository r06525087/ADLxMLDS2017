wget -O "model_final/model.ckpt-700.data-00000-of-00001" "https://www.dropbox.com/s/co597ezo1ls20an/model.ckpt-700.data-00000-of-00001?dl=0"
python hw2_seq2seq.py "$1" "$2" "$3"
