import torch
from torchvision.models.resnet import resnet50

import models_mae

dependencies = ["torch", "torchvision"]


def mae_vitb(pretrained=True, **kwargs):
    """ViT-Base pre-trained with MAE."""
    model = models_mae.__dict__["mae_vit_base_patch16"](**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def mae_vitl(pretrained=True, **kwargs):
    """ViT-Large pre-trained with MAE."""
    model = models_mae.__dict__["mae_vit_large_patch16"](**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def mae_vith(pretrained=True, **kwargs):
    """ViT-Huge pre-trained with MAE."""
    model = models_mae.__dict__["mae_vit_huge_patch16"](**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model
