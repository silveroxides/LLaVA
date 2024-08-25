import torch.nn as nn

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

    def forward(self, hidden_states):
        # Self-attention block
        residual = hidden_states
        
        # self attention
        hidden_states = self.self_attn(hidden_states)
        
        # add & nor
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        # Residual connection before the MoE block
        residual = hidden_states

        # MoE block
        hidden_states, router_Logits = self.moe(hidden_states)

        # Project the MoE output to match the residual dimension
        hidden_states = self.linear_projection(hidden_states)

        # Add the residual and apply the second layer normalization
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        outputs = (hidden_states, router_Logits)
        return outputs
