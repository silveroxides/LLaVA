import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimplifiedCrossAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        # Project inputs
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        # Final linear transformation
        output = self.output_linear(output)

        return output

class SimplifiedCrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimplifiedCrossAttentionLayer, self).__init__()
        self.cross_attention = SimplifiedCrossAttention(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, img_input, text_input):
        # Cross-attention
        img_cross = self.cross_attention(img_input, text_input, text_input)
        img = self.norm1(img_cross + img_input)  # Add & Norm

        # Feed-forward network
        img_ffn = self.ffn(img)
        img = self.norm2(img_ffn + img)  # Add & Norm

        return img

class SimplifiedCrossAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SimplifiedCrossAttentionEncoder, self).__init__()
        self.layers = nn.ModuleList([SimplifiedCrossAttentionLayer(input_dim, hidden_dim) 
                                     for _ in range(num_layers)])

    def forward(self, text_feature, visual_feature, text_mask=None, visual_mask=None):
        for layer in self.layers:
            visual_feature = layer(visual_feature, text_feature)
        return visual_feature
    
def get_co_attention(input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
    print(f'simplified cross attention implemented')
    return SimplifiedCrossAttentionEncoder(input_dim, hidden_dim, num_layers)


# # Example usage
# input_dim = 768
# hidden_dim = 2048
# num_layers = 2

# encoder = SimplifiedCrossAttentionEncoder(input_dim, hidden_dim, num_layers)
# visual_input = torch.rand(32, 10, input_dim)  # (batch_size, num_visual_tokens, input_dim)
# text_input = torch.rand(32, 20, input_dim)    # (batch_size, num_text_tokens, input_dim)

# output = encoder(visual_input, text_input)
# print(output.shape)  # Should be torch.Size([32, 10, 768])