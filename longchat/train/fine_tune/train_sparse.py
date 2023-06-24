# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from longchat.train.monkey_patch.llama_sparse import (
    replace_llama_with_sparse
)

from longchat.train.fine_tune.train import train

if __name__ == "__main__":
    replace_llama_with_sparse()
    train()
