
import torch
import csv
from PIL import Image
import numpy as np
from torchvision import transforms
import categories

DATA_DIR = "../TorchCNN/butterfly-data"

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, is_train=True):
        # Make dictionary that maps a category string to a number
        cat_dict = {category: i for i, category in enumerate(categories.categories)}

        # Make a transformer that will convert PIL images into
        # normalized tensors with shape (3, 224, 224)
        if is_train:
            self.data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            cvsname = "Training_set.csv"
            prefix = "train"
        else:
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            cvsname = "Validation_set.csv"
            prefix = "validate"

        # Read the CSV file putting the image paths and their labels (an integer)
        # in lists
        path = f"{DATA_DIR}/{cvsname}"
        with open(path, "r") as file:
            image_list = []
            label_list = []
            reader = csv.reader(file)
            # Skip the header row
            next(reader)
            for row in reader:
                filename = row[0].strip()
                category = row[1].strip()
                image_list.append(f"{DATA_DIR}/{prefix}/{filename}")
                label_list.append(cat_dict[category])

        # We will store the filenames and labels, but
        # we will load the actual images lazily to keep the memory
        # footprint reasonable
        self.x = image_list
        self.y = label_list

    # For a give index, look up the path, read in the image,
    # and transform to a tensor. Also return the label (an integer)
    def __getitem__(self, index):

        # Use pillow to open the appropriate image
        filename = self.x[index]
        img = Image.open(filename)
        if img.size != (224, 224) :
            print(f"*** {filename} size: {img.size} ***")

        # Resize, crop, and normalize
        img = self.data_transform(img)

        # Get the label
        label = self.y[index]

        # Return the tensor and the label
        return img, label

    def __len__(self):
        return len(self.x)
