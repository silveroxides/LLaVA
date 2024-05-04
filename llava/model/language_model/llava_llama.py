#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# thisis inherating all the attributes of LlamaConfig
# and adding new attribute called model_type attribute to it
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    num_experts_per_tok = 1
    num_experts = 1

# this was called by LlavaLlamaForCausalLM
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    # config: LlamaConfig -> specifying the type of the config argument
    def __init__(self, config: LlamaConfig):
        # calling the constructor of the parent class LlavaMetaModel and LlamaModel and passing the config argument to it.
        super(LlavaLlamaModel, self).__init__(config)

# LlavaLlamaForCausalLM is a subclass of both LlamaForCausalLM and LlavaMetaForCausalLM.
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, model_args=None):

        config.num_experts = model_args['num_experts']
        config.num_experts_per_tok = model_args['num_experts_per_tok']
        print('#' * 30 + '--model_args--' + '#' * 30)
        print(model_args)
        print('#' * 80)
        print('#' * 30 + '--Config -> Before Replacing--' + '#' * 30)
        print(config)
        print('#' * 80)
        print('#' * 30 + '--MM Projector Type -> Before Replacing--' + '#' * 30)
        print(config.mm_projector_type)
        print('#' * 80)
        # Replace default mm_projector_type with specified in model_args
        # if model_args and hasattr(model_args, 'mm_projector_type'):
        #     config.mm_projector_type = model_args.mm_projector_type
        if model_args and 'mm_projector_type' in model_args:
            config.mm_projector_type = model_args['mm_projector_type']
                        
        super(LlamaForCausalLM, self).__init__(config)

        print('#' * 30 + '--Config -> After Replacing--' + '#' * 30)
        print(config)
        print('#' * 80)
        print('#' * 30 + '--MM Projector Type -> After Replacing--' + '#' * 30)
        print(config.mm_projector_type)
        print('#' * 80)

        self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.gate_logits = ()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                gate_logits
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        self.gate_logits += (gate_logits,)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
