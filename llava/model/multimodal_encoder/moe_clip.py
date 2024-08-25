import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from typing import Any, Optional, Tuple
from transformers import CLIPConfig


class ModifiedEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: CLIPConfig, sparseMoe, hidden_size):
        super().__init__(config)  # Initialize CLIPEncoderLayer's components
        
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

        # Call the parent CLIPEncoderLayer's forward to handle self-attention and MLP
        outputs = super().forward(
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions
        )

        # Extract the hidden states
        hidden_states = outputs[0]

        # Residual connection before the MoE block
        residual = hidden_states

        # MoE block
        hidden_states, router_Logits = self.moe(hidden_states)

        # Project the MoE output to match the residual dimension
        hidden_states = self.linear_projection(hidden_states)

        # Add the residual and apply the second layer normalization
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (outputs[1],) # Add attention weights

        # Return the modified hidden states and router logits
        return outputs, router_Logits 