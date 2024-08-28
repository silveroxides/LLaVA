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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pprint

from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPConfig, CLIPVisionModel
from .multimodal_encoder.builder import build_vision_tower
from .co_attention.cross_attention import get_co_attention
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.mm_projector_type = config.mm_projector_type

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        # print('Inside initialize_vision_modules')
        # vision_tower = openai/clip-vit-large-patch14
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        share_moe = model_args.share_moe
        co_attention = model_args.cross_attention

        # preparing vision projection
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.num_experts = getattr(model_args, 'num_experts', 1)
        self.config.num_experts_per_tok = getattr(model_args, 'num_experts_per_tok', 1)
        self.config.aux_loss_coef = getattr(model_args, 'aux_loss_coef', 0.01)
        # gettting the config for the vision tower
        vision_tower_config = CLIPConfig.from_pretrained(vision_tower)
        self.config.mm_hidden_size = vision_tower_config.vision_config.hidden_size
        self.config.mm_vision_tower = vision_tower

        if getattr(self, 'mm_projector', None) is None:
            # print('-' * 100)
            # print('*'*40+'build viison projector'+'*'*40)

            self.mm_projector = build_vision_projector(self.config)
            # self.mm_projector = sparseMoE
            # print(self.mm_projector)
            # print('-'*120)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if self.get_vision_tower() is None:
            
            if share_moe:
                vision_tower = build_vision_tower(model_args, self.mm_projector)

            else: vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower

        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.hidden_size = self.config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.num_layers = getattr(model_args, 'num_layers', 2)
        self.num_heads = getattr(model_args, 'num_heads', 2)

        # initializing the co_attention
        if co_attention:
            self.co_attention = get_co_attention(self.hidden_size, self.config.intermediate_size, num_layers=self.num_layers, num_heads=self.num_heads, dropout_rate=0.1)
            

        if pretrain_mm_mlp_adapter is not None:
            print('using pretrain mlp adapter')
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            
            if self.config.mm_projector_type =='sparse_moe':
                print('initialize moe using pretrain mlp adapter')
                for i in range(model_args.num_experts):
                    self.mm_projector.experts[i].load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            else:
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


    def get_cross_attention(self):
        
        cross_attention = getattr(self, 'co_attention', None)

        if cross_attention is not None:
            return True
        else: return False


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_cross_attention(self):
        return self.get_model().get_cross_attention()
    
    def cross_attention(self, text_features, image_features, text_mask):
        return self.get_model().co_attention(image_features, text_features, text_mask, visual_mask = None)

    def encode_images(self, images):
        
        output_features = self.get_model().get_vision_tower()(images)

        try:
            # Try to unpack image_features, assuming it contains two values
            image_features, gate_logits_encoder = output_features
            # Process image_features if unpacking was successful
            image_features = self.get_model().mm_projector(image_features)
            

        except ValueError:
            # If unpacking fails, only image_features is returned and gate_logits_encoder should be None
            gate_logits_encoder = None
            image_features = output_features
            image_features = self.get_model().mm_projector(image_features)


        try:
            # Attempt to unpack the processed image_features again, assuming it could yield two values
            image_features, gate_logits = image_features

        except ValueError:
            # If unpacking fails, gate_logits should be None
            gate_logits = None

        # Return the appropriate output based on whether gate_logits_encoder is None
        if gate_logits_encoder is None:
            return image_features, gate_logits
        else:
            return image_features, gate_logits, gate_logits_encoder

        
    def pad_text_features(self, text_embeds):
        
        # Pad the text embeddings to the maximum length in the batch
        max_text_length = max(embed.size(0) for embed in text_embeds)
        padded_text_embeds = []
        
        for text_embed in text_embeds:
            padded_text_embed = torch.zeros((max_text_length, text_embed.size(1)), dtype=text_embed.dtype, device=text_embed.device)
            padded_text_embed[:text_embed.size(0)] = text_embed
            # print(f'text embed: {text_embed.shape}')
            # print(f'padded text embed: {padded_text_embed.shape}')
            padded_text_embeds.append(padded_text_embed)
        
        padded_text_features = torch.stack(padded_text_embeds)
        # attention_mask = text_embeds.sum(dim=-1) != 0

        
        return padded_text_features
    
    def remove_padding(self, text_features, attention_mask):
        """Removes padding from a batch of text features based on attention masks.

        Args:
            text_features (torch.Tensor): Batch of text features 
                (shape: batch_size x max_sequence_length x feature_dim).
            attention_mask (torch.Tensor): Batch of attention masks 
                (shape: batch_size x max_sequence_length), 
                where True indicates a valid token and False indicates padding.

        Returns:
            list: A list of tensors, each representing a sequence without padding.
        """

        unpadded_features = []
        for seq_features, seq_mask in zip(text_features, attention_mask):
            valid_indices = seq_mask.nonzero().squeeze()
            unpadded_features.append(seq_features[valid_indices])
        return unpadded_features
    
    def clip_contrastive_loss(self, text_embeddings, image_embeddings, attention_mask, temperature=0.07):

        # print(f'Text embeding shape: {text_embeddings.shape}')
        # print(f'Image embeding shape: {image_embeddings.shape}')
        # # convert this to fp32 to mitigate `nan` during normalization
        text_embeds = text_embeddings.float()
        vision_embeds = image_embeddings.float()

        # Normalize the embeddings
        normalized_text_embeds = F.normalize(text_embeds, dim=-1)  # Normalize across the embed_dim
        normalized_vision_embeds = F.normalize(vision_embeds, dim=-1)  # Normalize across the embed_dim

        # Create a mask for non-zero vectors
        attention_mask = attention_mask.float()

        # mean vision embeddings
        mean_vision_embeds = normalized_vision_embeds.mean(dim=1)  # [batch_size, embed_dim]
        mean_text_embeddings = normalized_text_embeds.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        # Compute the cosine similarity between all pairs
        logits_per_image = torch.matmul(mean_vision_embeds, mean_text_embeddings.T)  # [batch_size, batch_size]
        logits_per_text = logits_per_image.T  # [batch_size, batch_size]

        # Temperature parameter
        logits_per_image = logits_per_image / temperature
        logits_per_text = logits_per_text / temperature

        # print(f"Similarity matrix:\n{F.softmax(logits_per_image)}")

        # Ground truth labels
        batch_size = text_embeddings.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)

        # Compute the cross-entropy loss
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)

        # Total loss
        total_loss = (loss_image + loss_text) / 2
        
        # return total_loss.half()
        return total_loss
        


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None):
        
        vision_tower = self.get_vision_tower()
        cross_attention  = self.get_cross_attention()

        gate_logits = None
        align_loss = None
        gate_logits_encoder = None

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, gate_logits, align_loss, gate_logits_encoder

        
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            result = self.encode_images(concat_images)

            if len(result) == 3:
                image_features, gate_logits, gate_logits_encoder = result

            elif len(result) == 2:
                image_features, gate_logits = result
                
                if isinstance(gate_logits, tuple):
                    gate_logits_encoder = gate_logits
                    gate_logits = None
                else:
                    gate_logits_encoder = None

            elif len(result) == 1:
                image_features = result
                gate_logits = None
                gate_logits_encoder = None

            else:
                raise ValueError("Unexpected return value from encode_images.")
            
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        
        else:
            # Image Feature shape: torch.Size([4, 256, 5120]) -> [batch_size, sequence_length, embed_dim]
            result = self.encode_images(images)

    
            if len(result) == 3:
                image_features, gate_logits, gate_logits_encoder = result

            elif len(result) == 2:
                image_features, gate_logits = result
                
                if isinstance(gate_logits, tuple):
                    gate_logits_encoder = gate_logits
                    gate_logits = None
                else:
                    gate_logits_encoder = None

            elif len(result) == 1:
                image_features = result
                gate_logits = None
                gate_logits_encoder = None

            else:
                raise ValueError("Unexpected return value from encode_images.")
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # shape: torch.size[batch_size, sequence]
        # Shape of Inout_ids: torch.Size([4, 22])
        # Shape of attension_mask: torch.Size([4, 22])
        # Shape of position_ids: torch.Size([4, 22])
        # Shape of labels: torch.Size([4, 22])

        # remove the padding using attention_mask
        _input_ids = input_ids
        # inputs_ids =   [101, 2001, 2002, 2003, 0, 0, 0] -->   [101, 2001, 2002, 2003]
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # for ids in input_ids:
        #     print(f'Input ids length: {len(ids)}')
         
        
        new_input_embeds = []
        new_labels = []
        text_features = []
        splits = []
        text_labels = []
        cur_image_idx = 0

        # input_ids = [batch_size, sequence]
        # will pick one sequence from batch at a time
        for batch_idx, cur_input_ids in enumerate(input_ids):

            # print(f'[BEFORE] current_input_ids size: {len(cur_input_ids)}')
            # pprint.pprint(cur_input_ids)
            # pprint.pprint(labels[batch_idx])

            # getting sum of number of images present in given input_ids
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print('*'*+100)
            # print(f'Number of image: {num_images}')
            # print('*'*+100)
            # print(f'num_image: {num_images}')

            if num_images == 0:
                
                # print('*'*+100)
                # print('NO IMAGE')
                # print('*'*+100)

                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                # print(cur_input_embeds)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                text_features.append(cur_input_embeds)
                text_labels.append(labels[batch_idx])
                splits.append('')
                continue

            # cur_input_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  
            # labels = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])  

            # suppose 0, 4, 9 and 12 are the image tokens

            # Output: [-1, 3, 7, 22] (start, image positions, sequence_size) -> in this sequence 3rd and 7th token has the image token
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # image_token_indices = torch.tensor([-1, 0, 4, 9, 12])  

            # print(f'image_token_indices: {image_token_indices}')
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            for i in range(len(image_token_indices) - 1):

                start = image_token_indices[i] + 1 
                end = image_token_indices[i + 1]

                # Extract the segment (exclusive of the image token):
                cur_input_ids_noim.append(cur_input_ids[start:end])
                cur_labels_noim.append(cur_labels[start:end])

                # cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                # cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            # cur_input_ids_noim: [tensor([1, 2, 3]), tensor([5, 6, 7, 8]), tensor([10, 11])] 
            # cur_labels_noim:   [tensor([21, 22, 23]), tensor([25, 26, 27, 28]), tensor([30, 31])]

            # Stores the lengths of each text segment in cur_input_ids_noim/cur_labels_noim
            split_sizes = [x.shape[0] for x in cur_labels_noim] # split_sizes:  ([3, 4, 2] in our example).
            
            # concat the segments in a single tensor [tensor([1, 2, 3]), tensor([5, 6, 7, 8]), tensor([10, 11])] to
            # torch.cat(cur_input_ids_noim) --> tensor([ 1, 2, 3, 5, 6, 7, 8, 10, 11])
            # next we get the text feature for this input ids.
            # the length of input ids for our example is 9
            # after self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) -> it becomes shape of [9, embed_dimension]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))

            # Finally, the concatenated embeddings are split back into their original segments
            # from [9, embed_dimension] -> [3, embed_dimension], [4, embed_dimension], [2, embed_dimension] 
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            text_features.append(cur_input_embeds)
            text_labels.append(cur_labels_noim)
            splits.append(split_sizes)

            
            if cross_attention != True:

                # print('inside cross attension not true')
                
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                
                # print('*'*100)
                # for i in range(len(cur_new_input_embeds)):
                #     print(f'shape of index: {i} is {cur_new_input_embeds[i].shape}')
                # print('*'*100)

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)

                # print('*'*100)
                # print(f'current new input embeds: {cur_new_input_embeds.shape}')
                # print('*'*100)
                cur_new_labels = torch.cat(cur_new_labels)
                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
            
            # print('*'*100)
            # for i in range(len(text_features)):
            #     print(f'Shape of index {i} of text_features is: {text_features[i].shape}')
            # print('*'*100)
        # ##################################################### calculate the contrastive loss  #####################################################
        # for txt_feature in text_features:
        #     print(f' Shape of text_feature of: {txt_feature.shape}')
            
        text_features = [x.to(self.device) for x in text_features]

        padded_text_features = self.pad_text_features(text_features)
        # print(f'padded text feature shape: {padded_text_features.shape}')
        padded_text_features_attention_mask = padded_text_features.sum(dim=-1) != 0
        # Create the mask with the same dtype and device as the input
        padded_text_features_attention_mask =  padded_text_features_attention_mask.to(dtype=attention_mask.dtype, device=attention_mask.device)
        
        
        if cross_attention:

            image_features_has_zero = (image_features == 0).any().item()
            if image_features_has_zero:
                print('img feature got zero before passing to cross attn')

            # image_features, co_text_features = self.cross_attention(padded_text_features, image_features, padded_text_features_attention_mask)
            image_features, co_text_features = self.cross_attention(padded_text_features, image_features, padded_text_features_attention_mask)

            # text_features = self.remove_padding(co_text_features, padded_text_features_attention_mask)



            # Check for NaN values
            text_features_has_nan = torch.isnan(co_text_features).any().item()
            image_features_has_nan = torch.isnan(image_features).any().item()

            # Check for infinity values
            image_features_has_inf = torch.isinf(image_features).any().item()
            text_features_has_inf = torch.isinf(co_text_features).any().item()

            # Check for zero values
            image_features_has_zero = (image_features == 0).any().item()

            if text_features_has_nan:
                print("text_features_has_nan Contains NaN")
            if text_features_has_inf:
                print("text_features_has_inf Contains Inf")
            if image_features_has_nan:
                print("image_features_has_nan Contains NaN")
            if image_features_has_inf:
                print("image_features_has_inf Contains Inf:")

            
            for x in co_text_features:
                text_features_has_zero = (x == 0).any().item()
                if text_features_has_zero:
                    print("text_features_has_zero Contains Zero:", text_features_has_zero)

            if image_features_has_zero:
                print("image_features_has_zero Contains Zero:", image_features_has_zero)
            
            
            
            align_loss = self.clip_contrastive_loss(padded_text_features, image_features, padded_text_features_attention_mask)

            # print('unpad text features')
            # for i in text_features:
            #     print(i.shape)



            for x in range(len(text_features)):
                cur_input_embeds_no_im = text_features[x]
                # print(cur_input_embeds_no_im.shape)
                cur_labels_noim = text_labels[x]
                split_sizes = splits[x]
                # print(split_sizes)
                cur_input_embeds_no_im = torch.split(cur_input_embeds_no_im, split_sizes, dim=0)

                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                
                # print('*'*100)
                # for i in range(len(cur_new_input_embeds)):
                #     print(f'shape of index: {i} is {cur_new_input_embeds[i].shape}')
                # print('*'*100)

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)

                # print('*'*100)
                # print(f'current new input embeds: {cur_new_input_embeds.shape}')
                # print('*'*100)
                cur_new_labels = torch.cat(cur_new_labels)
                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

        else:
            align_loss = self.clip_contrastive_loss(padded_text_features, image_features, padded_text_features_attention_mask)


        # ##########################################################################################################################################################################

        # suppose
        # new_input_embeds: list of 4 tensors, each [seq_len, 5120] where seq_len varies (267, 264, 277, 269)
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        # print('*'*100)
        # print(f'Tokenizer model max length: {tokenizer_model_max_length}')
        # print('*'*100)
        if tokenizer_model_max_length is not None:
            
            # Truncate to max length (2048 in this case)
            # This operation truncates the tensors in new_input_embeds and new_labels to a maximum length of 2048.
            # it basically lining up the input ids within max length
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds] 
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # Combine them
        # find max length after truncating 
        # (267, 264, 277, 269) -> max_length = 277
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds) #(267, 264, 277, 269) -> batch size = 4

        # empty new_input_embeds
        new_input_embeds_padded = []
        
        # new_labels_padded dimensions: [batch_size, max_length]
        # filled with ignore index = -100
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)

        # new attention_mask dimensions: [batch_size, max_length]
        # filled with zeros        
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        # new position_ids dimensions: [batch_size, max_length]
        # filled with zeros         
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # iterate over the each current embeddings in the batch
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            
            # current length of the seqeunce -> suppose this one has shape [267, 5120] -> [seq_length, embed]
            cur_len = cur_new_embed.shape[0]

            # Pad the current input embeddings tensor to the maximum length: left or right padding according to the code
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0)) # [cur_len, embed_dim] -> [max_length, embed_dim] -> [267, 5120] ->  [277, 5120]

                # Fill the corresponding slice of new_labels_padded, attention_mask, and position_ids
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # all will be in same sequence length
        # in the batch the max length was 277 we are getting that
        #(267, 5120),  # [seq_len=267, 5120] -> [277, 5120] 
        # (264, 5120),  # [seq_len=264, 5120] -> [277, 5120] 
        # (277, 5120),  # [seq_len=277, 5120] -> [277, 5120] 
        # (269, 5120),  # [seq_len=269, 5120] -> [277, 5120] 
        
        # print('*'*100)
        # print('Padded Attension Mask shape: {attention_mask.shape}')
        # pprint.pprint(attention_mask)
        # print('*'*100)


        # checking with the actual passed parameters
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, gate_logits, align_loss, gate_logits_encoder

    
    # invoked from train.py
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
