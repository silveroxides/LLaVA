import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

class ModifiedEncoderLayer(nn.Module):
    def __init__(self, original_layer, hidden_size, sparseMoe):
        super().__init__()
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.layer_norm1 = original_layer.layer_norm1
        self.layer_norm2 = original_layer.layer_norm2

        # Initialize the Sparse MoE block and linear projection layer
        self.moe = sparseMoe
        self.linear_projection = nn.Linear(sparseMoe.experts[0][2].out_features, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        # MoE block
        hidden_states, router_Logits = self.moe(hidden_states)

        # Project the MoE output to match the residual dimension
        hidden_states = self.linear_projection(hidden_states)

        # Add the residual and apply the second layer normalization
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        outputs = (hidden_states, router_Logits)
        
        return outputs
