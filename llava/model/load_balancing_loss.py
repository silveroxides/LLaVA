import torch
from typing import Optional
from torch import nn
from torch.nn import functional as F

def aux_loss(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:

    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # got a tuple of len n
    # each of them have a shape of (batch_size, seq_length, num_experts)
    # upon checking the tuple we will just concat them in the o dimension
    # we will get (n*batch_size, sequence_Length, num_experts)
    if isinstance(gate_logits, tuple):
        # cat along the layers?
        compute_device = gate_logits[0].device
        gate_logits = torch.cat([gate.to(compute_device) for gate in gate_logits], dim=0) # (n*batch_size, sequence_Length, num_experts) -> type: tensor
        print(f'gate_logits: {gate_logits.shape}')

    _, selected_experts = torch.topk(gate_logits, top_k, dim=-1)


    routing_weights = gate_logits.softmax(dim=-1)


    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)


    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)

    # each row is a steps that correspondens to the fractions(percentage) of tokens in that steps experts has processed
    # [0.3333, 0.6667, 0.6667, 0.3333],
    # this one saying that expert one has process 33% of tokens the steps this assosiate with. simillarly experts 2, 3, and 4 has processed 66%, 66%, and 33% of tokens respectively!!!! 
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    router_prob_per_group_and_expert = torch.mean(routing_weights, dim=1)

    overall_loss = torch.sum(tokens_per_group_and_expert * router_prob_per_group_and_expert) * num_experts

    return overall_loss



def load_balancing_loss_func( gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
                            ) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    print('#'*100)
    # print(f'attention_mask: {attention_mask.shape}')

    print('-gate_logits-')
    for layer_idx, layer_logits in enumerate(gate_logits):
        print(f"Layer {layer_idx} shape: {layer_logits.shape}")
    print('#'*100)

    if gate_logits is None or not isinstance(gate_logits, tuple):
        print("gate_logits is None or not a tuple")
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        print(f'tokens_per_expert: {tokens_per_expert.shape}')

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
        print(f'router_prob_per_expert: {router_prob_per_expert.shape}')
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts