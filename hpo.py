import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import argparse
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device):
    """Evaluate the model's performance on the test set."""
    model.eval()
    total_loss = 0
    total_correct = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(preds == labels.data).item()

    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss


def train(model, data_loaders, num_epochs, criterion, optimizer, device):
    """Training loop for the neural network."""
    
    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in data_loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)


def net():
    """Initialize a pre-trained ResNet50 model and modify the final layer."""
    num_classes = 133
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_data_loaders(data, batch_size):
    """Prepare data loaders for training, validation, and testing."""
    return {
        split: torch.utils.data.DataLoader(data[split], batch_size=batch_size, shuffle=True)
        for split in ['train', 'valid', 'test']
    }


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    transformations = {
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

    image_data = {
        split: datasets.ImageFolder(os.path.join(args.data_dir, split), transformations[split])
        for split in ['train', 'valid', 'test']
    }

    data_loaders = create_data_loaders(image_data, args.batch_size)

    train(model, {'train': data_loaders['train'], 'valid': data_loaders['valid']}, 
          args.epochs, criterion, optimizer, device)
    
    avg_loss = test(model, data_loaders['test'], criterion, device)
    
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])