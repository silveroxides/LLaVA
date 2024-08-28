import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPConfig
from .clip_moe import CLIPSMoEVisionTransformer
from .moe_clip import ModifiedEncoderLayer
from .router_logits_collector import LogitCollectorWrapper




class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, sparseMoE=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.mlp_type = getattr(args, 'mm_projector_type', 'linear')
        self.num_experts = getattr(args, 'num_experts', 'linear')
        self.num_selected = getattr(args, 'num_experts_per_tok', 'linear')

        # check if sparse_Moe is none
        self.moe = sparseMoE is not None
        

        if not delay_load:
            if sparseMoE is not None: self.load_model(sparseMoE)
            else: self.load_model(sparseMoE)
            
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            if sparseMoE is not None: self.load_model(sparseMoE)
            else: self.load_model(sparseMoE)
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)


    def load_model(self, sparseMoE=None, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # vision encoder with moe
        if sparseMoE is not None:
            cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPSMoEVisionTransformer(cfg_only, sparseMoE, self.num_experts, self.num_selected)
            hidden_size = self.vision_tower.config.hidden_size
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            
            for i, encoder_layer in enumerate(self.vision_tower.vision_model.encoder.layers):
                self.vision_tower.vision_model.encoder.layers[i] = ModifiedEncoderLayer(encoder_layer, hidden_size, sparseMoE)

            print('shared moe encoder initialized')

            # Wrap the model with the LogitCollectorWrapper
            self.wrapped_vision_tower = LogitCollectorWrapper(self.vision_tower)

            # backnone freezing
            self.vision_tower.requires_grad_(False)

            for layer in self.wrapped_vision_tower.model.vision_model.encoder.layers:
                if isinstance(layer, ModifiedEncoderLayer):
                    for param in layer.moe.parameters():
                        param.requires_grad = True

                    for param in layer.linear_projection.parameters():
                        param.requires_grad = True

            # for name, param in self.wrapped_vision_tower.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name} is unfrozen")
                # else:
                #     print(f"{name} is frozen")
        
        # vanilla vision encoder
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.vision_tower.requires_grad_(False)
            print('pretrained vision model initialized')


    

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        router_logits = None

        # image is a list
        # for video
        if type(images) is list:
            image_features = []
            # router_logits = []
            
            if self.moe is None:
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
                    # router_logits.append(router_logits)

            else:
                for image in images:
                    image_forward_out = self.wrapped_vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_features = self.feature_select(image_forward_outs).to(images.dtype)
                    # print(f'image_features shape : {image_features.shape}')
                    image_features.append(image_feature)
                    # router_logits.append(router_logits)

        
        # image is not a list but tensor
        # for image
        else:
            
            if self.moe: 
                image_forward_outs = self.wrapped_vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
                router_logits = self.wrapped_vision_tower.get_collected_logits()
                
                # Iterate through the tuple to print the shapes of the logits
                # for i, logits in enumerate(router_logits):
                #     print(f"Layer {i} logits shape: {logits.shape}")
                # Clear logits if needed
                self.wrapped_vision_tower.clear_logits()

            
            else: 
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

            
            
        # Prepare the return tuple
        if router_logits is not None:
            return image_features, router_logits
        else:
            return image_features      

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, sparseMoE, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        # optional
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        print('*'*40+'vision config'+'*'*40)
        print(self.cfg_only)

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)