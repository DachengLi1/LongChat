#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python generate.py --model-name-or-path mosaicml/mpt-7b-storywriter --benchmark ghomasHudson/muld --dataset VLSP
