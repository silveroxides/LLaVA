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
import wandb

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..load_balancing_loss import *
from ..multimodal_encoder.clip_text_encoder import CustomTextEncoder

# this is inherating all the attributes of LlamaConfig
# and adding new attribute called model_type attribute to it
class LlavaConfig(LlamaConfig):
    # add new attributes to the config
    model_type = "llava_llama"


# this was called by LlavaLlamaForCausalLM
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    # config: LlamaConfig -> specifying the type of the config argument
    def __init__(self, config: LlamaConfig):
        # calling the constructor of the parent class LlavaMetaModel and LlamaModel and passing the config argument to it.
        super(LlavaLlamaModel, self).__init__(config)

# we have created this LlavaLlamaForCausalLM (child class) is a subclass of both LlamaForCausalLM (parent class) and LlavaMetaForCausalLM.
# LlamaForCausalLM is the main class which is inharited by LlavaLlamaForCausalLM
# now we can access all the functions of the parent class
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM,):
    # print('inside LlavaLlamaForCausalLM')
    config_class = LlavaConfig

    def __init__(self, config):

        # overriding default MRO as LlavaMetaForCausalLM does not have any _init_ function
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.config = config
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.use_custom_embed_tokens = False
        
        # Conditional replacement of the embedding layer
        if self.use_custom_embed_tokens:
            self.text_encoder = "openai/clip-vit-large-patch14"
            self.model.embed_tokens = CustomTextEncoder(self.text_encoder, config.hidden_size)

        # self.gate_logits = None
        self.gate_logits = [] # tuple of gate logits for each steps
        self.gate_logits_encoder = [] # tuple of gate logits for each steps
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
                gate_logits,
                alignment_loss,
                gate_logits_encoder
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

            if gate_logits is not None:
                self.gate_logits.append(gate_logits.cpu().detach())

            if gate_logits_encoder is not None:
                self.gate_logits_encoder.append(gate_logits_encoder)

        # self.gate_logits = (gate_logits,) # tuple of gate logits for each layer
        # self.gate_logits = gate_logits # tuple of gate logits for each layer
        # self.all_gate_logits += (gate_logits,) # tuple of gate logits for each layer
        # self.constrastive_loss = C_loss


        out = super().forward(
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
        
        if self.config.training:
            projector_type = getattr(self.config, 'mm_projector_type', 'linear')

            if projector_type == 'sparse_moe':
                load_balancing_loss = aux_loss(
                gate_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                ) * self.config.aux_loss_coef

                if gate_logits_encoder is not None:
                    encoder_moe_loss = load_balancing_loss_func(
                        gate_logits_encoder, 
                        self.config.num_experts,
                        self.config.num_experts_per_tok,
                        )* self.config.aux_loss_coef

                    load_balancing_loss+=encoder_moe_loss
                    if self.config.local_rank == 0:
                        print(f'encoder_moe_loss: {encoder_moe_loss}')


        
                llm_loss = out['loss']
                out['loss'] = llm_loss + load_balancing_loss.to(llm_loss.device) + alignment_loss.to(llm_loss.device)


                if self.config.local_rank == 0:
                    print(f'LLM Loss: {llm_loss}; LoadBalancingLoss: {load_balancing_loss}; AlignmentLoss: {alignment_loss}')
                    print(f'Total Loss: {out["loss"]}')

                    wandb.log({
                    "llm_loss": llm_loss,
                    "load_balancing_loss": load_balancing_loss,
                    "alignment_loss": alignment_loss,
                    # ... log any other metrics you want (e.g., accuracy) ... 
                    })

            else:
                llm_loss = out['loss']
                out['loss'] = llm_loss + alignment_loss.to(llm_loss.device)


                if self.config.local_rank == 0:
                    print(f'LLM Loss: {llm_loss}; AlignmentLoss: {alignment_loss}')
                    print(f'Total Loss: {out["loss"]}')

                    wandb.log({
                        "llm_loss": llm_loss,
                        "alignment_loss": alignment_loss,
                        # ... log any other metrics you want (e.g., accuracy) ... 
                    })
            

        return out

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
                _,
                _,
                _,
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
