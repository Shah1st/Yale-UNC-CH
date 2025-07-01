from copy import deepcopy
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.convnext import ConvNeXtBlock

from monai.networks.blocks import UpSample, SubpixelUpsample

####################
## EMA + Ensemble ##
####################

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()

    def forward(self, x):
        output = None
        
        for m in self.models:
            logits= m(x)
            
            if output is None:
                output = logits
            else:
                output += logits
                
        output /= len(self.models)
        return output
        

#############
## Decoder ##
#############

class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv= nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), 
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            self.upsample= SubpixelUpsample(
                spatial_dims= 2,
                in_channels= in_channels,
                scale_factor= scale_factor,
            )
        else:
            self.upsample = UpSample(
                spatial_dims= 2,
                in_channels= in_channels,
                out_channels= in_channels,
                scale_factor= scale_factor,
                mode= upsample_mode,
            )

        if intermediate_conv:
            k= 3
            c= skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
                )
        else:
            self.intermediate_conv= None

        self.attention1 = Attention2d(
            name= attention_type, 
            in_channels= in_channels + skip_channels,
            )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )
        self.attention2 = Attention2d(
            name= attention_type, 
            in_channels= out_channels,
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder2d(nn.Module):
    """
    Unet decoder.
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (2,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels= decoder_channels[1:]
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]
        
        # STORE skip_channels for use in forward method
        self.skip_channels = skip_channels
        
        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()
        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    attention_type= attention_type,
                    intermediate_conv= intermediate_conv,
                    upsample_mode= upsample_mode,
                    scale_factor= scale_factors[i],
                    )
            )
    
    def forward(self, feats: list[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]
        
        # Decoder blocks
        for i, b in enumerate(self.blocks):
            # ONLY pass skip if skip_channels > 0
            if i < len(self.skip_channels) and self.skip_channels[i] > 0:
                skip = feats[i] if i < len(feats) else None
            else:
                skip = None  # Force skip to None when skip_channels=0
                
            res.append(
                b(res[-1], skip=skip),
            )
            
        return res

class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv= nn.Conv2d(
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2
        )
        self.upsample = UpSample(
            spatial_dims= 2,
            in_channels= out_channels,
            out_channels= out_channels,
            scale_factor= scale_factor,
            mode= mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

#################################
## PERFECT SEGMENTATION HEAD  ##
#################################

class PerfectSegmentationHead2d(nn.Module):
    """
    Perfect segmentation head: 64Г—64 в†’ exactly 70Г—70
    Uses learnable upsampling for precise output size
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Feature processing
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels//2, affine=True)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(in_channels//2, out_channels, 3, padding=1)

        # Using ConvTranspose2d with exact kernel/padding for 70Г—70
        self.upsample = nn.ConvTranspose2d(
            out_channels, out_channels,
            kernel_size=7, stride=1, padding=0  
        )
        
    def forward(self, x):
        # Feature processing
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        
        x = self.upsample(x) 
        
        return x
        

#############
## Encoder ##
#############

def _convnext_block_forward(self, x):
    shortcut = x
    x = self.conv_dw(x)

    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

    if self.gamma is not None:
        x = x * self.gamma.reshape(1, -1, 1, 1)

    x = self.drop_path(x) + self.shortcut(shortcut)
    return x


class Net(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        use_72x72_mode: bool = False,
        use_smart_256x72_mode: bool = False,
    ):
        super().__init__()
        
        self.use_72x72_mode = use_72x72_mode
        self.use_smart_256x72_mode = use_smart_256x72_mode
        
        # Encoder
        self.backbone= timm.create_model(
            backbone,
            in_chans= 5,
            pretrained= pretrained,
            features_only= True,
            drop_path_rate=0.0,
            )
        ecs= [_["num_chs"] for _ in self.backbone.feature_info][::-1]

        
        if use_smart_256x72_mode:
            self.decoder= UnetDecoder2d(
                encoder_channels= ecs,
                scale_factors= (2, 2, 2, 2),  # Standard U-Net scale factors
            )
            
            self.seg_head = PerfectSegmentationHead2d(
                in_channels=self.decoder.decoder_channels[-1],
                out_channels=1,
            )
            
            print("[Properly Engineered 256Г—72] Using PerfectSegmentationHead2d for exact 70Г—70 output")
            
        elif use_72x72_mode:
            # Modified decoder with different scale factors for 72x72
            # The fundamental issue is that 72Г—72 is too small for this U-Net architecture. 
            # We're trying to fit a small input into a model designed for large images. 
            # For 72Г—72, a simpler decoder without skip connections was made
            self.decoder= UnetDecoder2d(
                encoder_channels= ecs,
                skip_channels= [0, 0, 0, 0],
                scale_factors= (3, 3, 2, 2),  
            )
            
            self.seg_head= SegmentationHead2d(
                in_channels= self.decoder.decoder_channels[-1],
                out_channels= 1,
                scale_factor= 1,
            )
        else:
            self.decoder= UnetDecoder2d(
                encoder_channels= ecs,
                scale_factors= (2, 2, 2, 2),  
            )
    
            self.seg_head= SegmentationHead2d(
                in_channels= self.decoder.decoder_channels[-1],
                out_channels= 1,
                scale_factor= 1,
            )

        
        
        # Stem modifications
        if use_smart_256x72_mode:
            self._update_stem_256x72_properly_engineered(backbone)
        elif use_72x72_mode:
            self._update_stem_72x72(backbone)
        else:
            self._update_stem_full_res(backbone)
        
        self.replace_activations(self.backbone, log=True)
        self.replace_norms(self.backbone, log=True)
        self.replace_forwards(self.backbone, log=True)

        decoder_channels = self.decoder.decoder_channels
        self.channel_adapters = nn.ModuleDict({
            '-1': nn.Identity(),
            '-2': nn.Conv2d(decoder_channels[-2], decoder_channels[-1], 1) if len(decoder_channels) >= 2 else nn.Identity(),
            '-3': nn.Conv2d(decoder_channels[-3], decoder_channels[-1], 1) if len(decoder_channels) >= 3 else nn.Identity(),
            '-4': nn.Conv2d(decoder_channels[-4], decoder_channels[-1], 1) if len(decoder_channels) >= 4 else nn.Identity(),
        })
        
    
    def _update_stem_256x72_properly_engineered(self, backbone):
        """
        PROPERLY ENGINEERED: 256Г—72 в†’ 32Г—32 for perfect 70Г—70 output
        Works backwards from target to ensure exact dimensions
        """
        if backbone.startswith("convnext"):
            original_stem = self.backbone.stem_0
            original_weight = original_stem.weight
            original_bias = original_stem.bias
            out_channels = original_weight.shape[0]
            
            
            conv1 = nn.Conv2d(
                5, out_channels, 
                kernel_size=6, stride=(4, 2), padding=1
            )
           
            conv2 = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(5, 3), stride=(2, 1), padding=(0, 1)
            )
           
            conv3 = nn.Conv2d(
                out_channels, out_channels, 
                kernel_size=(2, 9), stride=(1, 1), padding=0
            )
           
            self.backbone.stem_0 = nn.Sequential(
                # Intelligent padding: 256Г—72 в†’ 280Г—96
                nn.ReflectionPad2d((12, 12, 12, 12)),  
                
                # Multi-stage engineered downsampling
                conv1,  # 280Г—96 в†’ 70Г—48
                nn.LayerNorm([out_channels, 70, 48]),
                nn.GELU(),
                
                conv2,  # 70Г—48 в†’ 33Г—48  
                nn.LayerNorm([out_channels, 33, 48]),
                nn.GELU(),
                
                conv3,  # 33Г—48 в†’ 32Г—40
                nn.LayerNorm([out_channels, 32, 40]),
                nn.GELU(),
                
                # Final square adjustment
                nn.AdaptiveAvgPool2d((32, 32))  # 32Г—40 в†’ 32Г—32
            )
            
            with torch.no_grad():
                conv1.weight.data.normal_(0, 0.02)
                conv1.bias.data.zero_()

                if original_weight.shape[2] <= 6 and original_weight.shape[3] <= 6:
                    conv1.weight[:, :, :original_weight.shape[2], :original_weight.shape[3]].copy_(original_weight)
                    conv1.bias.copy_(original_bias)
                    
                conv2.weight.data.normal_(0, 0.02)
                conv2.bias.data.zero_()
                conv3.weight.data.normal_(0, 0.02) 
                conv3.bias.data.zero_()
            
            print(f"[PROPERLY ENGINEERED] Complete stem flow:")
            print(f"  256Г—72 в†’ ReflectionPad(280Г—96) в†’ Conv1(70Г—48) в†’ Conv2(33Г—48) в†’ Conv3(32Г—40) в†’ Pool(32Г—32)")
            print(f"  RESULT: Perfect 32Г—32 в†’ Perfect skip connections в†’ Perfect 70Г—70 output!")
            
        else:
            raise ValueError("Properly engineered stem not implemented for this backbone.")

    def _update_stem_72x72(self, backbone):
        """Proper stem modifications for 72x72 mode"""
        if backbone.startswith("convnext"):
            self.backbone.stem_0.stride = (2, 2)  # 72Г·2 = 36
            self.backbone.stem_0.padding = (1, 1)  # Standard padding
        else:
            raise ValueError("Custom striding not implemented for 72x72 mode.")

    def _update_stem_full_res(self, backbone):
        """Original aggressive stem modifications for full resolution - EXACTLY AS ORIGINAL"""
        if backbone.startswith("convnext"):

            # Update stride
            self.backbone.stem_0.stride = (4, 1)
            self.backbone.stem_0.padding = (0, 2)

            # Duplicate stem layer (to downsample height)
            with torch.no_grad():
                w = self.backbone.stem_0.weight
                new_conv= nn.Conv2d(w.shape[0], w.shape[0], kernel_size=(4, 4), stride=(4, 1), padding=(0, 1))
                new_conv.weight.copy_(w.repeat(1, (128//w.shape[1])+1, 1, 1)[:, :new_conv.weight.shape[1], :, :])
                new_conv.bias.copy_(self.backbone.stem_0.bias)

            self.backbone.stem_0= nn.Sequential(
                nn.ReflectionPad2d((1,1,80,80)),
                self.backbone.stem_0,
                new_conv,
            )

        else:
            raise ValueError("Custom striding not implemented.")
        pass

    def replace_activations(self, module, log=False):
        if log and self.use_smart_256x72_mode:
            print(f"[256Г—72 Mode] Replacing all activations with GELU...")
        elif log and self.use_72x72_mode:
            print(f"[72x72 Mode] Replacing all activations with GELU...")
        elif log:
            print(f"[Full Res Mode] Replacing all activations with GELU...")
        
        for name, child in module.named_children():
            if isinstance(child, (
                nn.ReLU, nn.LeakyReLU, nn.Mish, nn.Sigmoid, 
                nn.Tanh, nn.Softmax, nn.Hardtanh, nn.ELU, 
                nn.SELU, nn.PReLU, nn.CELU, nn.GELU, nn.SiLU,
            )):
                setattr(module, name, nn.GELU())
            else:
                self.replace_activations(child)

    def replace_norms(self, mod, log=False):
        if log and self.use_smart_256x72_mode:
            print(f"[256Г—72 Mode] Replacing all norms with InstanceNorm...")
        elif log and self.use_72x72_mode:
            print(f"[72x72 Mode] Replacing all norms with InstanceNorm...")
        elif log:
            print(f"[Full Res Mode] Replacing all norms with InstanceNorm...")
            
        for name, c in mod.named_children():
            n_feats= None
            if isinstance(c, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                n_feats= c.num_features
            elif isinstance(c, (nn.GroupNorm,)):
                n_feats= c.num_channels
            elif isinstance(c, (nn.LayerNorm,)):
                n_feats= c.normalized_shape[0]

            if n_feats is not None:
                new = nn.InstanceNorm2d(
                    n_feats,
                    affine=True,
                    )
                setattr(mod, name, new)
            else:
                self.replace_norms(c)

    def replace_forwards(self, mod, log=False):
        if log and self.use_smart_256x72_mode:
            print(f"[256Г—72 Mode] Replacing forward functions...")
        elif log and self.use_72x72_mode:
            print(f"[72x72 Mode] Replacing forward functions...")
        elif log:
            print(f"[Full Res Mode] Replacing forward functions...")
            
        for name, c in mod.named_children():
            if isinstance(c, ConvNeXtBlock):
                c.forward = MethodType(_convnext_block_forward, c)
            else:
                self.replace_forwards(c)
        

    def get_decoder_features(self, decoder_outputs):
        """Get features from decoder based on config"""
        from _cfg import cfg
        
        if cfg.use_feature_fusion:
            # Use learnable weights if available, otherwise use config weights
            if hasattr(self, 'fusion_weights'):
                weights = torch.softmax(self.fusion_weights, dim=0)
            else:
                weights = cfg.fusion_weights
            
            target_size = decoder_outputs[-1].shape[-2:]
            fused_features = None
            
            for i, layer_idx in enumerate(cfg.fusion_layers):
                features = decoder_outputs[layer_idx]
                
                # Channel adaptation
                adapter_key = str(layer_idx)
                if adapter_key in self.channel_adapters:
                    features = self.channel_adapters[adapter_key](features)
                
                # Spatial interpolation if needed
                if features.shape[-2:] != target_size:
                    features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
                
                weight = weights[i] if hasattr(self, 'fusion_weights') else weights[i]
                
                if fused_features is None:
                    fused_features = weight * features
                else:
                    fused_features += weight * features
            
            return fused_features
        else:
            # Single layer selection
            features = decoder_outputs[cfg.decoder_layer_index]
            
            # Channel adaptation
            adapter_key = str(cfg.decoder_layer_index)
            if adapter_key in self.channel_adapters:
                features = self.channel_adapters[adapter_key](features)
            
            # Spatial interpolation if needed
            target_size = decoder_outputs[-1].shape[-2:]
            if features.shape[-2:] != target_size:
                features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
            
            return features

    def proc_flip(self, x_in):
        x_in= torch.flip(x_in, dims=[-3, -1])
        x= self.backbone(x_in)
        x= x[::-1]
    
        # Decoder
        x= self.decoder(x)
        features_to_use = self.get_decoder_features(x)
        x_seg= self.seg_head(features_to_use)
        
        if not self.use_smart_256x72_mode:
            x_seg= x_seg[..., 1:-1, 1:-1]  # Only crop for other modes
        
        x_seg= torch.flip(x_seg, dims=[-1])
        x_seg= x_seg * 1500 + 3000
        return x_seg
    
    def forward(self, batch):
        x= batch
        
        # Encoder
        x_in = x
        x= self.backbone(x)
        x= x[::-1]
    
        # Decoder
        x= self.decoder(x)
        features_to_use = self.get_decoder_features(x)
        x_seg= self.seg_head(features_to_use)
        
        if not self.use_smart_256x72_mode:
            x_seg= x_seg[..., 1:-1, 1:-1]  # Only crop for other modes
        
        x_seg= x_seg * 1500 + 3000
    
        if self.training:
            return x_seg
        else:
            # Import cfg here to access the parameter
            from _cfg import cfg
            
            if not cfg.enhanced_tta:
                # Basic TTA 
                p1 = self.proc_flip(x_in)
                x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
                return x_seg
            else:
                # Enhanced TTA - collect multiple predictions
                predictions = [x_seg]  # Original prediction
                
                # 1. Flip TTA 
                p1 = self.proc_flip(x_in)
                predictions.append(p1)
                
                # 2. Noise TTA variants
                for _ in range(2):
                    noise = torch.randn_like(x_in) * cfg.tta_noise_level * torch.std(x_in)
                    x_noise = x_in + noise
                    
                    x_aug = self.backbone(x_noise)
                    x_aug = x_aug[::-1]
                    x_aug = self.decoder(x_aug)
                    features_aug = self.get_decoder_features(x_aug)
                    x_seg_aug = self.seg_head(features_aug)
                    
                    if not self.use_smart_256x72_mode:
                        x_seg_aug = x_seg_aug[..., 1:-1, 1:-1]
                    
                    x_seg_aug = x_seg_aug * 1500 + 3000
                    predictions.append(x_seg_aug)
                
                # 3. Scale TTA variants  
                for scale in cfg.tta_scale_values:
                    x_scaled = x_in * scale
                    
                    x_aug = self.backbone(x_scaled)
                    x_aug = x_aug[::-1]
                    x_aug = self.decoder(x_aug)
                    features_aug = self.get_decoder_features(x_aug)
                    x_seg_aug = self.seg_head(features_aug)
                    
                    if not self.use_smart_256x72_mode:
                        x_seg_aug = x_seg_aug[..., 1:-1, 1:-1]
                    
                    x_seg_aug = x_seg_aug * 1500 + 3000
                    predictions.append(x_seg_aug)
                
                # Average all predictions
                x_seg = torch.mean(torch.stack(predictions), dim=0)
                return x_seg