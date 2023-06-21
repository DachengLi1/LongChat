from typing import List, Optional, Tuple
import copy

import torch
from torch import nn

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm,apply_rotary_pos_emb, LlamaRotaryEmbedding

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

from local_attention import LocalAttention

from ..utils import rank0_print

def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel
    
    attention_mask: [bsz, q_len]
    """
    #rank0_print(self.config)
    sparse_pattern = getattr(self.config, "sparse_pattern", "")
    #rank0_print(f"Using sparse pattern {sparse_pattern}") 
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if sparse_pattern == "local":
        attn_fn = getattr(self, "attn_fn", None)
        if attn_fn is None:
            self.attn_fn = LocalAttention(window_size = 512, causal = True, autopad = True)
            rank0_print("Initializing local attention.")

        assert query_states.size() == (bsz, self.num_heads, q_len, self.head_dim), "Wrong query shape"
        attn_output = self.attn_fn(query_states, key_states, value_states)
        
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    else:
        # full attention (flash implementation)
        # [bsz, nh, t, hd]
        assert not output_attentions, "output_attentions is not supported"
        assert not use_cache, "use_cache is not supported"

        # Flash attention codes from
        # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

        # transform the data into the format required by flash attention
        qkv = torch.stack(
            [query_states, key_states, value_states], dim=2
        )  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
        # We have disabled _prepare_decoder_attention_mask in LlamaModel
        # the attention_mask should be the same as the key_padding_mask
        key_padding_mask = attention_mask

        if key_padding_mask is None:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            max_s = q_len
            cu_q_lens = torch.arange(
                0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
            )
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        else:
            nheads = qkv.shape[-2]
            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(
                x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
            )
            output_unpad = flash_attn_unpadded_qkvpacked_func(
                x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(
                pad_input(
                    rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
                ),
                "b s (h d) -> b s h d",
                h=nheads,
            )
        return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None


def model_init(self, config: LlamaConfig):
    super(LlamaModel, self).__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    # sparse information
    num_sparse_layers = config.num_hidden_layers - 2
    sparse_config = copy.deepcopy(config)
    sparse_config.sparse_pattern = "local"
    modules = [LlamaDecoderLayer(config)]
    for i in range(num_sparse_layers):
        modules.append(LlamaDecoderLayer(sparse_config))
    modules.append(LlamaDecoderLayer(config))
   # modules = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
    self.layers = nn.ModuleList(modules)
    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()

# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

def replace_llama_with_sparse():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = attn_forward
    transformers.models.llama.modeling_llama.LlamaModel.__init__ = model_init
