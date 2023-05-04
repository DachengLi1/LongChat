# Make attention memory and time efficient by monkey patching the LLaMA model with different kinds of efficient attention.

import sys
import argparse
# Need to call this before importing transformers.
from longchat.train.monkey_patch.llama_xformer_monkey_patch import (
    replace_llama_attn_with_xformer
)


if __name__ == "__main__":

    replace_llama_attn_with_xformer()

    from longchat.train.pretrain.pretrain import pretrain
    pretrain()

