from typing import List, Optional, Tuple
from functools import partial

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig

from einops import rearrange
from performer_pytorch import FastAttention

class PerformerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    """A performer version ported from: https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py."""
    def __init__(self, config: LlamaConfig, nb_features, redraw_interval):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        
        # Adopted from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
        # Currently Hardcoded nb_features=256 and redraw_interval=1000.
        # Remember to tune it. These two should influence performance a lot.
        self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = nb_features, causal = True)
        self.redraw_interval = redraw_interval
        self.count_since_last_redraw = 0

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        # Should be [bsz, nh, q_len, hd]
        assert query_states.size() == (bsz, self.num_heads, q_len, self.head_dim), "Wrong query shape"
        if self.redraw_interval != -1:
            self.count_since_last_redraw += 1
            if self.count_since_last_redraw % self.redraw_interval == 0:
                self.count_since_last_redraw = 0
                self.attn_fn.redraw_projection_matrix(query_states.device)

    #    print(query_states.shape, key_states.shape, value_states.shape)
    #    print(self.attn_fn)
        attn_output = self.attn_fn(query_states, key_states, value_states)

#        if attention_mask is not None:
#            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                raise ValueError(
#                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                )
#            attn_weights = attn_weights + attention_mask
#            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
    #    print(attn_output.shape)
        attn_output = attn_output.transpose(1, 2)
    #    print(attn_output.shape)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # Performer does not explicitely compute attention weight, so we can not output it.
        assert not output_attentions, "Performer cannot output attention scores"
        attn_weights = None

        return attn_output, attn_weights, past_key_value

def replace_llama_attn_with_performer(nb_features=128, redraw_interval=1000):
    transformers.models.llama.modeling_llama.LlamaAttention = partial(PerformerAttention, nb_features=nb_features, redraw_interval=redraw_interval) #.forward = forward
