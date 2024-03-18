import os, sys
import cv2
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform= transform)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle= shuffle)

    def __iter__(self):
        return iter(self.dataloader)
    def __len__(self):
        return len(self.dataloader)

### Find the original size of the images
for img in os.listdir('hands_archive/Train/A'):
    img = cv2.imread(f'hands_archive/Train/A/{img}')
print('Image shape: ', img.shape)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # RResize images to 224x224 while keeping the number of channels intact
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    transforms.RandomRotation(10),         # Randomly rotate the image by up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),        # Convert image to tensor
])

# Define the paths to your train and test directories
train_dir = 'hands_archive/Train/'
test_dir = 'hands_archive/Test/'


# Create train and test datasets using CustomImageDataset
train_dataset = CustomDataset(train_dir, transform=transform)
test_dataset = CustomDataset(test_dir, transform=transform)


for img_name in os.listdir('hands_archive/Train/A'):
    img_path = f'hands_archive/Train/A/{img_name}'
    img = cv2.imread(img_path)
    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_transformed = transform(image_pil)
print('Image shape: ', img_transformed.shape)


# Create train and test data loaders using CustomDataLoader
train_loader = CustomDataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = CustomDataLoader(test_dataset, batch_size=4, shuffle=False)

# # Iterate over batches in train_loader
# for batch_idx, (images, labels) in enumerate(train_loader):
#     # Accessing individual items within the batch
#     for image, label in zip(images, labels):
#         print(image.shape, label)

# Initialize an empty set to store unique labels
unique_labels = set()
# Iterate over the dataset to collect unique labels
for image, label in train_dataset:
    unique_labels.add(label)

# Get the total number of unique labels
total_labels = len(unique_labels)

print("Total number of unique labels:", total_labels)