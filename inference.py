
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.nn as nn



def model_fn(model_dir):

    model = net()
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def net():
    nclasses = 133
    
    # Pretrained model restnet50
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_inputs = model.fc.in_features
    
    model.fc = nn.Linear(num_inputs, nclasses)
    
    return model