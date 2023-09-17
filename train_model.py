import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets as torchvision_datasets
import smdebug.pytorch as smd
import argparse

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse


def evaluate_model(model, eval_loader, loss_fn, device, debug_hook=None):
    model.eval()

    if debug_hook:
        debug_hook.set_mode(smd.modes.EVAL)

    total_loss = 0
    total_corrects = 0

    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
        
        _, pred_labels = torch.max(predictions, 1)
        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(pred_labels == labels.data).item()

    avg_loss = total_loss / len(eval_loader.dataset)
    avg_accuracy = total_corrects / len(eval_loader.dataset)
    
    print(f"Evaluation Accuracy: {100 * avg_accuracy}%, Loss: {avg_loss:}")
    return avg_accuracy, avg_loss

def train_model(model, data_loaders, num_epochs, loss_fn, optim, device, debug_hook=None):
    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                if debug_hook:
                    debug_hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                if debug_hook:
                    debug_hook.set_mode(smd.modes.EVAL)

            running_loss, running_corrects = 0.0, 0

            for images, labels in data_loaders[phase]:
                images, labels = images.to(device), labels.to(device)
                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optim.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - {phase} - Loss: {epoch_loss}, Accuracy: {100*epoch_acc}%")

    return model

def initialize_model():
    num_classes = 133
    pretrained_model = models.resnet50(pretrained=True)
    for param in pretrained_model.parameters():
        param.requires_grad = False
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
    return pretrained_model

def load_trained_model(model_path):
    model = initialize_model()
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def data_loaders_from_dataset(dataset, batch_size=64):
    return {split: torch.utils.data.DataLoader(dataset[split], batch_size, shuffle=True) for split in ['train', 'valid', 'test']}

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model().to(device)

    debug_hook = smd.Hook.create_from_json_file()
    debug_hook.register_hook(model)

    loss_fn = nn.CrossEntropyLoss()
    debug_hook.register_loss(loss_fn)

    optimizer = optim.Adam(model.fc.parameters(), args.lr)

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

    datasets = {split: torchvision_datasets.ImageFolder(os.path.join(args.data_dir, split), transformations[split]) for split in ['train', 'valid', 'test']}
    data_loaders = data_loaders_from_dataset(datasets, args.batch_size)

    train_model(model, {'train': data_loaders['train'], 'valid': data_loaders['valid']}, args.epochs, loss_fn, optimizer, device, debug_hook)
    evaluate_model(model, data_loaders['test'], loss_fn, device, debug_hook)
    
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, metavar="E", help="number of epochs to train (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", ""))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_DATA", ""))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', ""))
    args = parser.parse_args()

    main(args)
