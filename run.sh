#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --dataset qasper &
CUDA_VISIBLE_DEVICES=1 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --dataset qmsum &
CUDA_VISIBLE_DEVICES=2 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --dataset narrative_qa &
CUDA_VISIBLE_DEVICES=3 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --dataset gov_report &
