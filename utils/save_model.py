import torch
from torchvision import models

def save_model(model):
    torch.save(model.state_dict(), 'trained_model_weight.pth')