import matplotlib.pyplot as plt
import torch
from torch import nn

import torchvision
from torchvision import datasets, models, transforms
from tiny_vgg import TinyConvNet

from path import Path
import kaggle
import os
import zipfile
import gc

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

def data_transforms():
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

def get_food101_dataset(location='PyTorch', download=True):
    """
    The function to download the Food-101 dataset and split into train, validation, and test sets.
    The only argument is to specify whether to download from Kaggle, or from the PyTorch library.
    """
    if location == 'Kaggle':
        # Make sure you have your API token file at the correct folder
        # Follow https://www.kaggle.com/docs/api#authentication
        path = Path('kmader/food41')
        if not path.exists():
            kaggle.api.dataset_download_cli(str(path))
        