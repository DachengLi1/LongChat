from typing import List, Optional, Tuple
from functools import partial

import torch
from torch import nn

import transformers
#from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig

from einops import rearrange

# TODO: Understand layer_head_mask
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    attention_window: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    hidden_states = hidden_states.transpose(0, 1)

    query_states = self.q_proj(hidden_states)#.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states)#.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states)#.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    
    query_states /= math.sqrt(self.head_dim) # Move normalization to here
    # TODO(Dacheng): Longformer does a seemingly useless transpose in the HF implementation. Use it for now.
    query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).transpose(0,1)
    key_states = key_states.view(q_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

    kv_seq_len = key_states.shape[-2]

    assert past_key_value is None and not use_cache, "Does not support caching for now"
    #if past_key_value is not None:
    #    kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    #if past_key_value is not None:
        # reuse k, v, self_attention
    #    key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #    value_states = torch.cat([past_key_value[1], value_states], dim=2)

    #past_key_value = (key_states, value_states) if use_cache else None

    ### Begin Longformer code ###
    one_sided_attn_window_size = attention_window // 2
    attn_weights = _sliding_chunks_query_key_matmul(
            query_states, key_states, one_sided_attn_window_size
    )

    # TODO(Dacheng): Deal with attention mask so that they are multiple of window size outside...
            # values to pad for attention probs
    remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

    # cast to fp32/fp16 then replace 1's with -inf
    float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
    )
    # diagonal mask with zeros everywhere and -inf inplace of padding
    diagonal_mask = _sliding_chunks_query_key_matmul(
        float_mask.new_ones(size=float_mask.size()), float_mask, one_sided_attn_window_size
    )

    # pad local attention probs
    attn_weights += diagonal_mask

    # Note this is different than usual output shape, especially dim1 and dim2are reverse.
    assert list(attn_weights.size()) == [
        bsz,
        q_len,
        self.num_heads,
        one_sided_attn_window_size * 2 + 1,
    ], (
        f"local_attn_probs should be of size ({bsz}, {q_len}, {self.num_heads},"
        f" {one_sided_attn_window_size * 2 + 1}), but is of size {attn_weights.size()}"
    )
    #if attention_mask is not None:
    #    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #        raise ValueError(
    #            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
    #    attn_weights = attn_weights + attention_mask
    #    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    #attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # TODO: (Dacheng) Understand what this is 
    #softmax sometimes inserts NaN if all positions are masked, replace them with 0
    #attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)

    attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)  # use fp32 for numerical stability


    value_states = value_states.view(q_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
    
    attn_output = _sliding_chunks_matmul_attn_probs_value(
            attn_weights, value_states, one_sided_attn_window_size
    )
    assert attn_output.size() == (bsz, q_len, self.num_heads, self.head_dim), "Unexpected size"
    attn_output = attn_output.transpose(0, 1).reshape(q_len, bsz, query_states.shape[-1]).contiguous()
    attn_output = (attn_output.transpose(0, 1),)
  #  attn_output = torch.matmul(attn_weights, value_states)

#    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#        raise ValueError(
#            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#            f" {attn_output.size()}"
#        )

#    attn_output = attn_output.transpose(1, 2)
#    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def _sliding_chunks_matmul_attn_probs_value(
    self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
):
    """
    Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
    same shape as `attn_probs`
    """
    batch_size, seq_len, num_heads, head_dim = value.size()

    assert seq_len % (window_overlap * 2) == 0
    assert attn_probs.size()[:3] == value.size()[:3]
    assert attn_probs.size(3) == 2 * window_overlap + 1
    chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
    # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

    chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        batch_size * num_heads,
        torch.div(seq_len, window_overlap, rounding_mode="trunc"),
        window_overlap,
        2 * window_overlap + 1,
    )

    # group batch_size and num_heads dimensions into one
    value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

    # pad seq_len with w at the beginning of the sequence and another window overlap at the end
    padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

    # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
    chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    chunked_value_stride = padded_value.stride()
    chunked_value_stride = (
        chunked_value_stride[0],
        window_overlap * chunked_value_stride[1],
        chunked_value_stride[1],
        chunked_value_stride[2],
    )
    chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

    chunked_attn_probs = _pad_and_diagonalize(chunked_attn_probs)

    context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)


def _sliding_chunks_query_key_matmul(query: torch.Tensor, key: torch.Tensor, window_overlap: int):
    """
    Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
    implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
    overlap of size window_overlap
    """
    batch_size, seq_len, num_heads, head_dim = query.size()
    assert (
        seq_len % (window_overlap * 2) == 0
    ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
    assert query.size() == key.size()

    chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

    # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
    query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

    query = _chunk(query, window_overlap)
    key = _chunk(key, window_overlap)

    # matrix multiplication
    # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
    # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
    # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
    diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply

    # convert diagonals into columns
    diagonal_chunked_attention_scores = _pad_and_transpose_last_two_dims(
        diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
    )

    # allocate space for the overall attention matrix where the chunks are combined. The last dimension
    # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
    # window_overlap previous words). The following column is attention score from each word to itself, then
    # followed by window_overlap columns for the upper triangle.

    diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    )

    # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        :, :, :window_overlap, : window_overlap + 1
    ]
    diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        :, -1, window_overlap:, : window_overlap + 1
    ]
    # - copying the lower triangle
    diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
    ]

    diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        :, 0, : window_overlap - 1, 1 - window_overlap :
    ]

    # separate batch_size and num_heads dimensions again
    diagonal_attention_scores = diagonal_attention_scores.view(
        batch_size, num_heads, seq_len, 2 * window_overlap + 1
    ).transpose(2, 1)

    _mask_invalid_locations(diagonal_attention_scores, window_overlap)
    return diagonal_attention_scores

def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]
    ending_mask = beginning_mask.flip(dims=(1, 3))
    beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        beginning_input, -float("inf")
    ).where(beginning_mask.bool(), beginning_input)
    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_mask = ending_mask.expand(ending_input.size())
    input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        ending_input, -float("inf")
    ).where(ending_mask.bool(), ending_input)

def _pad_and_diagonalize(chunked_hidden_states):
    """
    shift every row 1 step right, converting columns into diagonals.
    Example:
    ```python
    chunked_hidden_states: [
        0.4983,
        2.6918,
        -0.0071,
        1.0492,
        -1.8348,
        0.7672,
        0.2986,
        0.0285,
       -0.7584,
        0.4206,
        -0.0405,
        0.1599,
        2.0514,
        -1.1600,
        0.5372,
        0.2629,
    ]
    window_overlap = num_rows = 4
    ```
                (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                  0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                   -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
    """
    total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
    chunked_hidden_states = nn.functional.pad(
        chunked_hidden_states, (0, window_overlap + 1)
    )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, -1
    )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
    chunked_hidden_states = chunked_hidden_states[
        :, :, :-window_overlap
    ]  # total_num_heads x num_chunks x window_overlap*window_overlap
    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
    )
    chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    return chunked_hidden_states


def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
    """pads rows and then flips rows and columns"""
    hidden_states_padded = nn.functional.pad(
        hidden_states_padded, padding
    )  # padding value is not important because it will be overwritten
    hidden_states_padded = hidden_states_padded.view(
        *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
    )
    return hidden_states_padded

def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
        window_overlap * 2,
        hidden_states.size(2),
    )
    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1
    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

def replace_llama_attn_with_longformer():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
