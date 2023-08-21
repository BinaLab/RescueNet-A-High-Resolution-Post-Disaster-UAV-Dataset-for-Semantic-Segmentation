from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from models.vit import VisionTransformer
from models.utils import checkpoint_filter_fn
from models.decoder import DecoderLinear
from models.decoder import MaskTransformer
from models.segmenter import Segmenter
#import segm.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    #model_cfg = model_cfg.copy()
    backbone = model_cfg.backbone

    normalization = model_cfg.normalization
    #model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    d_ff = mlp_expansion_ratio * model_cfg.d_model

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg.train_h,
        model_cfg.train_h,
    )
    image_size = (model_cfg.train_h, model_cfg.train_w)
    model = VisionTransformer(image_size,
        model_cfg.patch_size,
        model_cfg.encoder_n_layers,
        model_cfg.d_model,
        d_ff,
        model_cfg.n_heads,
        1000)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    #decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.decoder_name 
    d_encoder = decoder_cfg.d_model
    #decoder_cfg["d_encoder"] = decoder_cfg.d_model
    #decoder_cfg["patch_size"] = decoder_cfg.patch_size

    
    if name == "mask_transformer":
        dim = decoder_cfg.d_model
        n_heads = dim // 64
        #decoder_cfg["n_heads"] = n_heads
        #decoder_cfg["d_model"] = dim
        d_ff = 4 * dim
        decoder = MaskTransformer(decoder_cfg.classes, decoder_cfg.patch_size, d_encoder,
                    decoder_cfg.decoder_n_layers, decoder_cfg.n_heads, decoder_cfg.d_model, d_ff,
                    decoder_cfg.drop_path_rate, decoder_cfg.dropout)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    
    """
    if "linear" in name:
        decoder = DecoderLinear(decoder_cfg)
    elif name == "mask_transformer":
        dim = decoder_cfg.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        d_ff = 4 * dim
        decoder = MaskTransformer(decoder_cfg.classes, decoder_cfg.patch_size, decoder_cfg.d_encoder,
                    decoder_cfg.n_layers, decoder_cfg.n_heads, decoder_cfg.d_model, d_ff, decoder_cfg.d_model,
                    decoder_cfg.drop_path_rate, decoder_cfg.dropout)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    """
    return decoder


def create_segmenter(model_cfg):
    #model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg
    #decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    #model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])
    model = Segmenter(encoder, decoder, n_cls=model_cfg.classes)

    return model

"""
def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
"""
