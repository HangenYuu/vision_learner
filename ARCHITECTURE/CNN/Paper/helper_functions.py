import torch
from torch import nn

from torchvision import datasets, models, transforms
from torchmetrics import Accuracy
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from path import Path
from sklearn.model_selection import train_test_split
import kaggle
import os
import zipfile
import json
import shutil
import gc

from tiny_vgg import TinyConvNet
from my_dataset import MyDataset

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute((1, 2, 0))
    
    # Undo preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = (std * image + mean)
    # Image needs to be clipped between 0 and 1
    # or it looks like noise when displayed
    # (Something I learnt from Udacity)
    image = torch.clip(image, 0, 1)
    
    ax.imshow(image)    
    return ax

def build_tiny_vgg(input_shape, hidden_units, output_shape, device):
    return TinyConvNet(input_shape, hidden_units, output_shape).to(device)

def ResNet18(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(nn.Linear(512, 256),
                         nn.ReLU(),
                         nn.Dropout(p=0.4),
                         nn.Linear(256, num_classes),
                         nn.LogSoftmax(dim=1))
    model = model.to(device)

    return model

def _data_transforms():
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(35),
                                      transforms.RandomVerticalFlip(0.27),
                                      transforms.RandomHorizontalFlip(0.27),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_n_test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transforms, valid_n_test_transforms

def get_food101_dataset_kaggle(batch_size=32, download=True):
    """
    The function to download the Food-101 dataset and split into train, validation, and test sets.
    The only argument is to specify whether to download from Kaggle, or from the PyTorch library.
    """
    # Make sure you have your API token file at the correct folder
    # Follow https://www.kaggle.com/docs/api#authentication
    if download:
        path = Path('kmader/food41')
        if not path.exists():
            kaggle.api.dataset_download_cli(str(path))
        data_path = Path('data')
        if not data_path.exists():
            os.mkdir(data_path)
        zipfile.ZipFile('food41.zip').extractall(data_path)
        with open('/content/food41/meta/meta/train.json', 'r') as fp:
            train_dict = json.load(fp)
        with open('/content/food41/meta/meta/test.json', 'r') as fp:
            test_dict = json.load(fp)
        if not os.path.exists('data'):
            os.mkdir('data')
        new_data_path = Path('data')
        original_data_path = Path('food41/images')
        new_folders = ['train', 'test']
        for folder in new_folders:
            if not os.path.exists(new_data_path/folder):
                os.mkdir(new_data_path/folder)
            if folder == 'train':
                if not os.path.exists(new_data_path/'valid'):
                    os.mkdir(new_data_path/'valid')
                for key, value in train_dict.items():
                    train_value, valid_value = train_test_split(value, train_size=0.75)
                    train_set, valid_set = set(train_value), set(valid_value)
                    if not os.path.exists(new_data_path/folder/key):
                        os.mkdir(new_data_path/folder/key)
                    if not os.path.exists(new_data_path/'valid'/key):
                        os.mkdir(new_data_path/'valid'/key)
                    for image in os.listdir(original_data_path/key):
                        image_path = key + '/' + image
                        image_path = image_path.split('.')[0]
                        if image_path in train_set:
                            shutil.copy(original_data_path/key/image, new_data_path/'train'/key/image)
                        if image_path in valid_set:
                            shutil.copy(original_data_path/key/image, new_data_path/'valid'/key/image)
            else:
                for key, value in test_dict.items():
                    value_set = set(value)
                    if not os.path.exists(new_data_path/folder/key):
                        os.mkdir(new_data_path/folder/key)
                    for image in os.listdir(original_data_path/key):
                        image_path = key + '/' + image
                        image_path = image_path.split('.')[0]
                        if image_path in value_set:
                            shutil.copy(original_data_path/key/image, new_data_path/folder/key/image)
        shutil.rmtree(original_data_path)
    new_data_path = Path('data')
    train_dir = new_data_path/'train'
    valid_dir = new_data_path/'valid'
    test_dir = new_data_path/'test'
    train_transforms, valid_n_test_transforms = _data_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_n_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = valid_n_test_transforms)
    class_names = test_dataset.classes
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, class_names

def get_food101_dataset_torch(batch_size=32, download=True):
    data_dir = Path('data')
    train_transforms, valid_n_test_transforms = _data_transforms()
    train_dataset = datasets.Food101(data_dir, transform=None, download=download)
    test_dataset = datasets.Food101(data_dir, split='test', transform=valid_n_test_transforms, download=download)
    lengths = [0.75, 0.25]
    train_subset, valid_subset = torch.utils.data.random_split(train_dataset, lengths)
    train_dataset = MyDataset(train_subset, transform=train_transforms)
    valid_dataset = MyDataset(valid_subset, transform=valid_n_test_transforms)
    class_names = test_dataset.classes
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, class_names

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metric: Accuracy,
               device: torch.device):
    train_loss, train_acc = 0, 0
    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss & accuracy
        loss = criterion(y_pred, y)
        train_loss += loss
        train_acc += metric(y_pred.argmax(dim=1), y)

        # 3. Empty out gradient
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Optimize 1 step
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               metric: Accuracy,
               device: torch.device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for (X,y) in data_loader:
            X, y = X.to(device), y.to(device)
        # 1. Forward pass
            y_pred = model(X)

        # 2. Calculate loss & accuracy
            test_loss += criterion(y_pred, y)
            test_acc += metric(y_pred.argmax(dim=1), y)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          data_loaders: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          metric: Accuracy,
          device: torch.device,
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           data_loaders[0],
                                           criterion,
                                           optimizer,
                                           metric,
                                           device)
        gc.collect()
        torch.cuda.empty_cache()

        test_loss, test_acc = test_step(model,
                                        data_loaders[1],
                                        criterion,
                                        metric,
                                        device)
        gc.collect()
        torch.cuda.empty_cache()
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss.cpu().detach().numpy())
        results["train_acc"].append(train_acc.cpu().detach().numpy())
        results["test_loss"].append(test_loss.cpu().detach().numpy())
        results["test_acc"].append(test_acc.cpu().detach().numpy())

    # 6. Return the filled results at the end of the epochs
    return results

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()