import torch
from torch import nn 
import torchvision


def create_ViT_model(num_class:int,
                     seeds:int=42):
  #GEt pretrained weights for ViT-Base
  pretrained_vit_weights=torchvision.models.ViT_B_16_Weights.DEFAULT  # Default = best avaialable

  # Getting automatic transforms from pretrained ViT weights
  vit_transforms=pretrained_vit_weights.transforms()

  # Setup a ViT model instance with pretrained weights
  pretrained_vit=torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

  # Freeze the base parameters
  for parameter in pretrained_vit.parameters():
    parameter.requires_grad=False

  # Updating the classifier head
  pretrained_vit.heads=nn.Linear(in_features=768,
                                out_features=num_class).to(device)
  return pretrained_vit,vit_transforms