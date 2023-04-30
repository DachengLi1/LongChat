# Make attention memory and time efficient by monkey patching the LLaMA model with different kinds of efficient attention.

import sys
import argparse
# Need to call this before importing transformers.
from llama_performer_monkey_patch import (
    replace_llama_attn_with_performer,
)
from llama_local_monkey_patch import (
    replace_llama_attn_with_local,
)


if __name__ == "__main__":

   # parser = argparse.ArgumentParser()
   # parser.add_argument('--attention_mode', type=str)
   # args = parser.parse_args()

   # if args.attention_mode == "performer":
   #     replace_llama_attn_with_performer(nb_features=128, redraw_interval=-1)
   # elif args.attention_mode == "local":
    replace_llama_attn_with_local()

    from train import train
    train()

