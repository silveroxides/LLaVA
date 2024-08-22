import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class CustomTextEncoder(nn.Module):
    def __init__(self, clip_model_name: str, llama_embedding_dim: int):
        super(CustomTextEncoder, self).__init__()
        self.clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.projection = nn.Linear(self.clip_text_encoder.config.hidden_size, llama_embedding_dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.clip_text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = outputs.last_hidden_state
        print(f'Text embedding dimension:{text_embeds.shape}')
        projected_embeds = self.projection(text_embeds)
        print(f'After projecting Text embedding dimension:{projected_embeds.shape}')
        return projected_embeds
