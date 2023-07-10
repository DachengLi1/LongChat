CUDA_VISIBLE_DEVICES=7 python eval_mmlu.py --model-name-or-path lmsys/longchat-13b-16k --max_seq_len 16000 &
CUDA_VISIBLE_DEVICES=6 python eval_mmlu.py --model-name-or-path lmsys/vicuna-13b-v1.3 --max_seq_len 16000 &
