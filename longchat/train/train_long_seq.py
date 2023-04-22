# Make attention memory and time efficient by monkey patching the LLaMA model with Performer attention.

import sys
# Need to call this before importing transformers.
from llama_performer_monkey_patch import (
    replace_llama_attn_with_performer,
)

replace_llama_attn_with_performer()

from train import train

if __name__ == "__main__":
    train()

