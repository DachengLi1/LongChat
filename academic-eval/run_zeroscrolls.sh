#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --benchmark tau/zero_scrolls --dataset gov_report --max_seq_len 83968 --max_new_tokens 1000 &
CUDA_VISIBLE_DEVICES=1 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --benchmark tau/zero_scrolls --dataset summ_screen_fd --max_seq_len 83968 --max_new_tokens 1000 &
CUDA_VISIBLE_DEVICES=2 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --benchmark tau/zero_scrolls --dataset qmsum --max_seq_len 83968 --max_new_tokens 1000 &
CUDA_VISIBLE_DEVICES=3 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --benchmark tau/zero_scrolls --dataset squality --max_seq_len 83968 --max_new_tokens 1000 &

python generate.py --model-name-or-path mosaicml/mpt-30b-chat --benchmark tau/zero_scrolls --dataset gov_report --num_gpus 8 --max_gpu_memory 10 --max_seq_len 8192 --max_new_tokens 1000
