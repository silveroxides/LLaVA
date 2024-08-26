import torch
import torch.nn as nn
from collections import defaultdict
from .moe_clip import ModifiedEncoderLayer

class LogitCollectorWrapper(nn.Module):
    def __init__(self, clip_vision_model):
        super().__init__()
        self.model = clip_vision_model
        self.logits_buffer = defaultdict(list)
        self._hook_layers()

    def _hook_layers(self):
        for i, layer in enumerate(self.model.vision_model.encoder.layers):
            if isinstance(layer, ModifiedEncoderLayer):
                layer.register_forward_hook(self._create_hook(i))

    def _create_hook(self, layer_idx):
        def hook(module, input, output):
            logits = module._last_router_logits
            if logits is not None:
                self.logits_buffer[layer_idx].append(logits.detach().cpu())
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_collected_logits(self):
        logits_list = [torch.stack(self.logits_buffer[layer_idx]) for layer_idx in sorted(self.logits_buffer.keys())]
        return tuple(logits_list)

    def clear_logits(self):
        self.logits_buffer.clear()