import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf')).type_as(scores)
        
        attention = self.dropout(F.softmax(scores, dim=-1))
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.dropout(output)
        return self.out(output)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(F.relu(self.dropout(self.linear1(x))))

class DualStreamLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(DualStreamLayer, self).__init__()
        self.self_attn_visual = MultiHeadAttention(input_dim, num_heads, dropout)
        self.self_attn_text = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn_visual = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn_text = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ff_visual = FeedForward(input_dim, hidden_dim, dropout)
        self.ff_text = FeedForward(input_dim, hidden_dim, dropout)
        self.norm1_visual = nn.LayerNorm(input_dim)
        self.norm2_visual = nn.LayerNorm(input_dim)
        self.norm3_visual = nn.LayerNorm(input_dim)
        self.norm1_text = nn.LayerNorm(input_dim)
        self.norm2_text = nn.LayerNorm(input_dim)
        self.norm3_text = nn.LayerNorm(input_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, visual_input, text_input, visual_mask=None, text_mask=None):
        # Self-attention for visual stream
        visual = self.self_attn_visual(visual_input, visual_input, visual_input, visual_mask)
        visual = self.norm1_visual(visual_input + visual)
        
        # Self-attention for text stream
        text = self.self_attn_text(text_input, text_input, text_input, text_mask)
        text = self.norm1_text(text_input + text)
        
        # Cross-attention
        visual_cross = self.cross_attn_visual(visual, text, text, text_mask)
        visual = self.norm2_visual(visual + visual_cross)
        
        text_cross = self.cross_attn_text(text, visual, visual, visual_mask)
        text = self.norm2_text(text + text_cross)
        
        # Feed-forward
        visual = self.norm3_visual(visual + self.ff_visual(visual))
        text = self.norm3_text(text + self.ff_text(text))
        
        return visual, text

class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(CrossAttentionModel, self).__init__()
        self.layers = nn.ModuleList([DualStreamLayer(input_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, visual_feature, text_feature, text_mask=None, visual_mask=None):
        
        visual_feature = self.dropout(visual_feature)
        text_feature = self.dropout(text_feature)
        # text_mask = text_mask.float()

        for layer in self.layers:
            visual_feature, text_feature = layer(visual_feature, text_feature, visual_mask, text_mask)
        return visual_feature, text_feature

 
def get_co_attention(input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
    return CrossAttentionModel(input_dim, hidden_dim, num_layers, num_heads, dropout_rate)
    
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