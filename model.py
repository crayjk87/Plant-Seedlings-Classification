import torch.nn as nn
from torchvision import models
import torch.optim as optim

def trans_resnet_50():

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 12)
    )

    return model
