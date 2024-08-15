import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        query_proj = self.query_linear(query)
        key_proj = self.key_linear(key)
        value_proj = self.value_linear(value)

        # Calculate attention scores
        scores = torch.matmul(query_proj, key_proj.transpose(-2, -1))
        
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, value_proj)
        return output
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Linear transformations for queries, keys, and values for each head
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        
        # Final linear transformation after concatenating heads
        self.out_linear = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear transformations for queries, keys, and values for each head
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf')).type_as(scores)
        
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads and perform final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_len, input_dim)
        output = self.out_linear(output)
        return output
    
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super(CrossAttentionLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.cross_attention = MultiHeadCrossAttention(input_dim, num_heads)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)        
        self.linear3 = nn.Linear(input_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, input_dim)
        self.norm1_visual = nn.LayerNorm(input_dim)
        self.norm2_visual = nn.LayerNorm(input_dim)
        self.norm3_visual = nn.LayerNorm(input_dim)
        self.norm1_text = nn.LayerNorm(input_dim)
        self.norm2_text = nn.LayerNorm(input_dim)
        self.norm3_text = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_input, text_input, text_mask=None, visual_mask=None):
        
        # Self-attention
        img, _ = self.self_attention(img_input, img_input, img_input)
        img = self.norm1_visual(img_input + img)

        text, _ = self.self_attention(text_input, text_input, text_input)
        text = self.norm1_text(text_input + text)

        # Cross-attention
        img_cross = self.cross_attention(img, text, text, text_mask)
        img = self.norm2_visual(img_cross + img)

        text_cross = self.cross_attention(text, img, img)
        text = self.norm2_text(text_cross + text)
        
        # Feed-forward networks with dropout after LayerNorm but before the last linear layer
        img_ffn = self.linear2(F.relu(self.norm3_visual(self.linear1(img))))
        img = img_ffn + img  # Adding residual connection

        text_ffn = self.linear4(F.relu(self.norm3_text(self.linear3(text))))
        text = text_ffn + text  # Adding residual connection

        img = self.dropout(img)  # Dropout after the residual connection
        text = self.dropout(text)  # Dropout after the residual connection        
 
        return img, text

class CrossAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout = 0.05):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(input_dim, hidden_dim, num_heads, dropout) 
                                     for _ in range(num_layers)])

    def forward(self, visual_feature, text_feature, text_mask=None, visual_mask=None):

        for layer in self.layers:
            visual_feature, text_feature = layer(visual_feature, text_feature, text_mask, visual_mask)
        
        return visual_feature, text_feature
    
def get_co_attention(input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
    return CrossAttentionEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout_rate)

    
# hudai = CrossAttentionEncoder(768, 768, 2, 2)
# x = torch.rand(32, 10, 768)
# memory = torch.rand(32, 20, 768)

# output = hudai(x, memory)
# output.shape

# multihead_attn = nn.MultiheadAttention(768, 2)
# cross = MultiHeadCrossAttention(768, 2)
# attn_output, attn_output_weights = multihead_attn(x, x, x)
# attn_output = cross(x, memory, memory)
# attn_output.shape