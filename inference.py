import os
import torch
import torch.nn as nn
from torchvision import models

def load_checkpoint(model, model_dir):
    """
    Load a model's state dictionary from a checkpoint.

    """
    checkpoint_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def initialize_model(num_classes=133, pretrained=True):
    """
    Initialize a pre-trained ResNet50 model and modify the final layer for classification.
    
    """
    # Load the pre-trained ResNet50 model
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze all layers in the network for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final layer for classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def model_fn(model_dir):
    """
    Load the saved model from the provided directory.

    """
    model = initialize_model()
    return load_checkpoint(model, model_dir)
