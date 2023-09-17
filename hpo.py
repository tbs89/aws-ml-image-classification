import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

def test(model, test_loader, criterion, device):
    """Evaluate model performance on test dataset."""
    model.eval()
    running_loss = 0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    
    print(f"Testing Loss: {total_loss:.4f}")


def train(model, train_loaders, epochs, criterion, optimizer, device):
    """Train the model using the provided data and parameters."""
    print("Starting training...")
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            running_loss = 0
            running_correct = 0
            
            if phase == 'train':
                model.train()
                print(f"Training epoch: {epoch+1}/{epochs}")
            else:
                model.eval()
                print(f"Validation epoch: {epoch+1}/{epochs}")

            for data, target in train_loaders[phase]:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * data.size(0)
                running_correct += torch.sum(preds == target).item()

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / len(train_loaders[phase].dataset)
            epoch_acc = running_correct / len(train_loaders[phase].dataset)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss}, Accuracy: {epoch_acc}")

    return model

def net():
    """Initialize and return a pre-trained model with customized final layer."""
    print("Initializing the network...")
    
    num_classes = 133
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_inputs = model.fc.in_features
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model

def create_data_loaders(data, batch_size):
    """Create and return dataloaders for train, valid, and test datasets."""
    print("Setting up data loaders...")
    
    dataloaders = {
        split: torch.utils.data.DataLoader(data[split], batch_size, shuffle=True)
        for split in ['train', 'valid', 'test']
    }
    return dataloaders

def main(args):
    print(f"Selected Hyperparameters:\nEpochs: {args.epochs}\nBatch Size: {args.batch_size}\nLearning Rate: {args.lr}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(args.data_dir, split), data_transforms[split])
        for split in ['train', 'valid', 'test']
    }

    dataloaders = create_data_loaders(image_datasets, args.batch_size)
    train_loaders = {
        'train': dataloaders['train'],
        'valid': dataloaders['valid']
    }

    model = train(model, train_loaders, args.epochs, loss_criterion, optimizer, device)
    test(model, dataloaders['test'], loss_criterion, device)
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)
    print("Model saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for NN")
    parser.add_argument("--epochs", type=int, default=1, metavar="E", help="number of epochs to train (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args = parser.parse_args()
    
    main(args)