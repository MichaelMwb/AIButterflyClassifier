import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from PIL import Image
import os
import categories
import pickle
from ImageDataset import ImageDataset

# Constants
MODEL_PATH = "model.pth"
DATA_DIR = "butterfly-data"

# Figure out what sort of device you will be computing on: "cpu", "cuda" or "mps"
device =torch.device("cuda" if torch.cuda.is_available() else "cpu") ## Your code here
print(f"Using {device} device.")

# Same transform we used in training
data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) ## Your code here

# How many categories are there?
category_count = len(categories.categories)

# Get the filenames in a list
filelist_path = f"{DATA_DIR}/Testing_set.csv"
filelist_file = open(filelist_path, "r")
filenames = [path.strip() for path in filelist_file]

# Delete the header
del filenames[0]
filecount = len(filenames)

# Read in the stored model, put it on your devices
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
model = model.to(device)
print(f"Loaded model from {MODEL_PATH} → model device: {next(model.parameters()).device}")

# Create the file we are writing the predictions to
outfile = open("predictions.csv", "w")
print("filename,prediction", file=outfile)

# Not training (disable dropout, turn off gradient bookkeeping)
model.eval()
with torch.no_grad():

    # Instead of batches, process the images one at a time
    for filename in filenames:
        path = f"{DATA_DIR}/test/{filename}"

        # Use pillow to load the image
        img = Image.open(path).convert("RGB")## Your code here

        # Use the transformer to resize, crop, and normalize
        img = data_transform(img)## Your code herer

        # Make it a batch of one
        inputs = img.unsqueeze(0).to(device)
        # Do inference
        outputs = model(inputs) ## Your code here

        # Get the hard prediction as a butterfly species name
        label = categories.categories[torch.argmax(outputs, dim=1).item()] ## Your code here

        # Store the filenane and prediction in a file
        print(f"{filename},{label}", file=outfile)

outfile.close()
