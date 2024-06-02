#    Copyright 2018- The Hugging Face team. All rights reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from CLIP (https://github.com/huggingface/transformers)
# Copyright 2024 Jiachen Li
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from typing import Dict, Optional, Sequence, List

from transformers.activations import ACT2FN
from einops import rearrange, repeat, reduce, pack, unpack

class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class CLIPEncoderMoELayer(nn.Module):
    def __init__(self, config, sparseMoE):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # self.num_of_experts = config.num_of_experts
        # self.num_selected = config.num_selected
        # self.gate = nn.Linear(self.embed_dim, self.num_of_experts, bias=False)
        # self.experts = nn.ModuleList([CLIPMLP(config) for _ in range(self.num_of_experts)])
        self.CLIPMoE = sparseMoE
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        gate_logits = self.gate(hidden_states)

        router_z_loss = torch.logsumexp(gate_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        
        gate_softmax = nn.functional.softmax(gate_logits, dim=-1, dtype=torch.float).to(hidden_states.dtype)

        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')

        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)

        one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_of_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_of_experts ** 2)

        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(hidden_states.dtype)
        
        results = torch.zeros_like(hidden_states).to(hidden_states.device, hidden_states.dtype)
        for b in range(hidden_states.shape[0]):
            for i, expert in enumerate(self.experts):
                token_idx, nth_expert = torch.where(selected_experts[b] == i)
                results[b][token_idx] += weights[b][token_idx, nth_expert, None] * expert(hidden_states[b][token_idx])
        #hidden_states = self.mlp(hidden_states)
        hidden_states = residual + results

        outputs = (hidden_states, balance_loss, router_z_loss)
        return outputs

class CLIPEncoder(nn.Module):
    def __init__(self, config, sparseMoE):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderMoELayer(config, sparseMoE) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        inputs_embeds
    ):
        encoder_states = ()
        hidden_states = inputs_embeds
        balance_losses = []
        router_z_losses = []
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]
            balance_loss = layer_outputs[1]
            balance_losses.append(balance_loss)
            router_z_loss = layer_outputs[2]
            router_z_losses.append(router_z_loss)
        return encoder_states, balance_losses, router_z_losses

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class CLIPSMoEVisionTransformer(nn.Module):
    def __init__(self, config, sparseMoE, num_experts=4, num_selected=2):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        config.num_of_experts = num_experts
        config.num_selected = num_selected
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config, sparseMoE)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs, balance_losses, router_z_losses = self.encoder(hidden_states)
        return encoder_outputs[-1], torch.stack(balance_losses).mean(), torch.stack(router_z_losses).mean()