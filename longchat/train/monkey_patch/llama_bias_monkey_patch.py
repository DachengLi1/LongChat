from transformers.models.llama import LlamaForCausalLM
from torch import nn

class LlamaForCausalLMBias(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._hidden_size = self.model.config.hidden_size
        self._intermediate_size = self.model.config.intermediate_size
        self._vocab_size = self.model.config.vocab_size
        self.add_bias()
        print(self.model)

    def add_bias(self):
        for layer in self.model.layers:
            # layer is a LlamaDecoderLayer
            self._add_bias_attn(layer.self_attn)
            self._add_bias_mlp(layer.mlp)
        self.model.lm_head = nn.Linear(in_features=self._hidden_size, out_features=self._vocab_size, bias=True)

    def _add_bias_attn(self, attn_module):
        attn_module.q_proj = nn.Linear(in_features=self._hidden_size, out_features=self._hidden_size, bias=True)
        attn_module.k_proj = nn.Linear(in_features=self._hidden_size, out_features=self._hidden_size, bias=True)
        attn_module.v_proj = nn.Linear(in_features=self._hidden_size, out_features=self._hidden_size, bias=True)
        attn_module.o_proj = nn.Linear(in_features=self._hidden_size, out_features=self._hidden_size, bias=True)
    
    def _add_bias_mlp(self, mlp_module):
        mlp_module.gate_proj = nn.Linear(in_features=self._hidden_size, out_features=self._intermediate_size, bias=True)
        mlp_module.down_proj = nn.Linear(in_features=self._intermediate_size, out_features=self._hidden_size, bias=True)
        mlp_module.up_proj = nn.Linear(in_features=self._hidden_size, out_features=self._intermediate_size, bias=True)
