We included commands to run evaluation reported in the [blog](https://lmsys.org/blog/2023-06-29-longchat/)
```bash
python3 eval.py --model-name-or-path  lmsys/longchat-7b-16k --task <task> --longchat_flash_attn
python3 eval.py --model-name-or-path  lmsys/longchat-13b-16k --task <task> --longchat_flash_attn
python3 eval.py --model-name-or-path  mosaicml/mpt-7b-storywriter --task <task>
python3 eval.py --model-name-or-path  mosaicml/mpt-30b-chat --task <task> --num_gpus 8 --max_gpu_memory 10
python3 eval.py --model-name-or-path  THUDM/chatglm2-6b --task <task>
```
