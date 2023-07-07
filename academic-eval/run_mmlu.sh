CUDA_VISIBLE_DEVICES=7 python eval_mmlu.py --model-name-or-path lmsys/longchat-13b-16k &
CUDA_VISIBLE_DEVICES=6 python eval_mmlu.py --model-name-or-path lmsys/vicuna-13b-v1.3 &
