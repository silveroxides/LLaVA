import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers.activations import ACT2FN

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
    

class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.mm_hidden_size #mm_hidden_size": 1024
        self.ffn_dim = config.hidden_size    #hidden_size": 4096
        self.num_experts = config.num_experts #num_experts": 4
        self.top_k = config.num_experts_per_tok #num_experts_per_tok": 2

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # self.experts = nn.ModuleList([Experts(config) for _ in range(self.num_experts)])
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)) for _ in range(self.num_experts)])


        # Jitter parameters
        self.jitter_noise = 0.01
        # self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # print('-' * 40 + '--Hidden States: Experts--' + '-' * 40)
        # print(f'Batch Size: {batch_size}, Sequence Length: {sequence_length}, Hidden Dim: {hidden_dim}')
        # # print(f'Hidden States Shape: {hidden_states.shape}')
        # print('-' * 100)
        # hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        # print(f'Shape of gate weights:", {tuple(self.gate.weight.shape)} - (num_experts, dim)')
        # print('#################################################################################################################################')           
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # print('#################################################################################################################################')
        # print(f'router_logits: {router_logits.shape} - (sequence_length, num_experts)')
        # print('#################################################################################################################################')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.ffn_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # print('#################################################################################################################################')
        # print(f'final_hidden_states: {final_hidden_states.shape}')
        # print('#################################################################################################################################')

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            # print(f'Expert: {expert_idx}')
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # print('#################################################################################################################################')
            # print(f'current_hidden_states Shape: {current_hidden_states.shape}')
            # print('#################################################################################################################################')

            '''
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            RuntimeError: source tensor shape must match self tensor shape, 
            excluding the specified dimension. Got self.shape = [18432, 1024] source.shape = [12869, 4096]
            '''

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # reshaping the final hiddle state from [*hidden_states] to [batch_size, sequence_length, self.ffn_dim]
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.ffn_dim)
        reshapeed_router_logits = router_logits.reshape(batch_size, sequence_length, self.num_experts)
        batch_size, sequence_length, hidden_dim = final_hidden_states.shape
        # print('-' * 40 + '--Final Hidden States--' + '-' * 40)
        # print(f'Final Hiddens states shape: {final_hidden_states.shape}')
        # print(f'Batch Size: {batch_size}, Sequence Length: {sequence_length}, Hidden Dim: {hidden_dim}')    
        # print('-' * 100)
        # print(f'reshapeed_router_logits: {reshapeed_router_logits.shape}')
        return final_hidden_states, router_logits
        # return final_hidden_states


    
class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        print('#' * 100)
        print('linear Projection executed in BUILDER.PY')
        print('#' * 100)
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == 'custom':
        return nn.Sequential(
                        nn.Linear(config.mm_hidden_size, config.mm_hidden_size*2),
                        nn.GELU(),
                        nn.Linear(config.mm_hidden_size*2, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, config.hidden_size)
                        )
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        print('#' * 100)
        print(f'{projector_type} Projection executed in BUILDER.PY')
        print('#' * 100)
        
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    if projector_type == 'sparse_moe':
        print('sparse_moe Projection executed in BUILDER.PY')
        print('-' * 100)
        return SparseMoeBlock(config)
    
    raise ValueError(f'Unknown projector type: {projector_type}')


