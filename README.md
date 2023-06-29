# LongChat
LongChat supports training and benchmarking long-context large language model based chatbots. The library is named longchat and longeval respectively.

## News
- [2023/06]🔥 We introduced LongChat models and the evaluation benchmark longeval.

## Contents
- [Install] (#install)
- [Training] (#longchat)
- [Model Weights] (#model-weights)
- [Evaluation] (#longeval)

## Install
```bash
conda create -n longeval python=3.10
conda activate longeval
git clone https://github.com/DachengLi1/LongChat/
cd LongChat/
pip install -e .
```
For users who want to test very long sequence length, also install flash-attention by:
```bash
git clone https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

## longchat
To train a LongChat model yourself, replace <path-to-llama> to the llama checkpoint director, and run:
```bash
python -m torch.distributed.run --nproc_per_node=8 \
         longchat/train/fine_tune/train_condense_16K.py \
        --model_name_or_path <path-to-llama> \
        --data_path data/dummy_conversation.json  \
        --bf16 \
        --output_dir outputs \
        --num_train_epochs 3    \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4  \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000  \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 16384  \
        --gradient_checkpointing True  \
        --lazy_preprocess True
```
This script assumes 8xA100 GPUs and use the dummy data in the repository for example usage only. Please adapt to your use case.

## Model Weights
Model Weights are available through HuggingFace Hub: [LongChat-13b-16k](https://huggingface.co/lmsys/longchat-13b-16k) and [LongChat-7b-16k](https://huggingface.co/lmsys/longchat-7b-16k).

## longeval
To evaluate the LongChat model on the coarsed-grained topics benchmark:
```bash
cd longeval
python3 eval.py --model-name-or-path  lmsys/longchat-13b-16k --task topics --longchat_flash_attn
```

More generally, to evaluate new models, simply choose a ```<task>``` from ["topics", "lines"], replace ```<your-model>``` with your model path, and run
```bash
python3 eval.py --model-name-or-path <your-model> --task <task>
```
Some models require memory efficient flash attention to evaluate super long test. We include the commands we used:
```bash
python3 eval.py --model-name-or-path  lmsys/longchat-7b-16k --task <task> --longchat_flash_attn
python3 eval.py --model-name-or-path  lmsys/longchat-13b-16k --task <task> --longchat_flash_attn
python3 eval.py --model-name-or-path  mosaicml/mpt-7b-storywriter --task <task>
python3 eval.py --model-name-or-path  mosaicml/mpt-30b-chat --task <task> --num_gpus 8 --max_gpu_memory 10
python3 eval.py --model-name-or-path  THUDM/chatglm2-6b --task <task> 
```
All experiments can be run on a single A100 40GB GPU except mpt-30b-chat.



