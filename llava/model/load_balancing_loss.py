import torch
from typing import Optional
from torch import nn
from torch.nn import functional as F

def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally apply the straight-through estimator.
    Args:
      logits: [batch_size, num_classes] unnormalized log-probs
      temperature: non-negative scalar
      hard: whether to apply the straight-through estimator
    Returns:
      [batch_size, num_classes] sample from the Gumbel-Softmax distribution
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-9))
    gumbel_dist = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(gumbel_dist, dim=-1)
    if hard:
        _, indices = torch.topk(y_soft, k=1, dim=-1)
        y_hard = torch.zeros_like(y_soft).scatter_(dim=-1, index=indices, value=1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

def aux_loss(gate_logits, num_experts, top_k=2, temperature=1.0, hard=True):
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Concatenate gate logits along the batch dimension
    gate_logits = torch.cat(gate_logits, dim=0)

    # Sample from the Gumbel-Softmax distribution and apply straight-through estimator
    routing_weights = gumbel_softmax(gate_logits, temperature=temperature, hard=hard)

    # Select top-k experts using the sampled routing weights
    _, selected_experts = torch.topk(routing_weights, k=top_k, dim=-1)

    # Compute the expert mask and mean number of tokens per expert
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()
    tokens_per_group_and_expert = expert_mask.mean(dim=-2)

    # Compute the routing probability per expert and mean routing probability per expert
    router_prob_per_group_and_expert = routing_weights.mean(dim=-2)
    mean_router_prob_per_expert = router_prob_per_group_and_expert.mean(dim=-2)

    # Compute the load balancing loss
    overall_loss = torch.sum(mean_router_prob_per_expert - tokens_per_group_and_expert) * num_experts

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