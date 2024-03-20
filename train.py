import os, sys
import cv2
import torchvision
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from readData import *

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print('device', device)

BATCH_SIZE = 4
LR = 0.0001
num_epochs = 1

# Define the paths to your train and test directories
train_dir = 'hands_archive/Train/'
test_dir = 'hands_archive/Test/'

# Create train and test datasets using CustomImageDataset
train_dataset = CustomDataset(train_dir, transform=transform)
test_dataset = CustomDataset(test_dir, transform=transform)

# Create train and test data loaders using CustomDataLoader
train_loader = CustomDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = CustomDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize an empty set to store unique labels
unique_labels = set()
# Iterate over the dataset to collect unique labels
for image, label in train_dataset:
    unique_labels.add(label)

# Get the total number of unique labels
num_classes = len(unique_labels)
print("Total number of unique labels:", num_classes)

model = torchvision.models.resnet50(pretrained= True)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)


for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)
        ## forward
        scores = model(data)
        loss = criterion(scores, target)
        losses.append(loss)

        ## Backward
        optimizer.zero_grad()
        loss.backward()

        ## gradient descend or adam step
        optimizer.step()
    print(f'cost at {epoch}, is {sum(losses)/len(losses)}')

