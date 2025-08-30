import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import os
import categories
import pickle
from ImageDataset import ImageDataset
from time import perf_counter

# Figure out what sort of device you will be computing on: "cpu", "cuda" or "mps"
if torch.cuda.is_available():
    device = torch.device("cuda")          # NVIDIA GPUs (Win/Linux)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")           # Apple Silicon (macOS)
else:
    device = torch.device("cpu")           # Everyone else
print(f"Using {device} device.")

# Constants
MODEL_PATH = "model.pth"

# You can set these to anything that works well
# for your system and your optimizer
NUM_EPOCHS = 24
BATCH_SIZE = 100

# Get ready to read training and validation data
training_data = ImageDataset(True)
validation_data = ImageDataset(False)
training_dataloader= torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
validation_dataloader= torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

# How many categories are there?
category_count = len(categories.categories)

# Have we stored the model?
if os.path.exists(MODEL_PATH):

    # Read in the stored model (not just the weights), put it on your device
    model = torch.load(MODEL_PATH)
    model.to(device)
    print(f"Loaded model from {MODEL_PATH}")

else:
    # Download torchvision's pretrained VGG16 model (IMAGENET1K_V1_
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

    # Print model summary
    summary(model, input_size=(3, 224, 224))

    # Move the model to the device
    model = model.to(device)

    # Replace the last fully connected layer with a new one that has the
    # correct number of output features, on your device
    # (The layer before has 4096 outputs)
    model.classifier[-1] = nn.Linear(4096, category_count).to(device)
    print("Model created using pretrained weights")

# Freeze all the layers
for params in model.parameters():
    params.requires_grad_(False)

# Except the fully connected layers
training_count = 0
training_parameters = []
for layer in model.classifier:
    for params in layer.parameters():
        training_count += params.numel()
        training_parameters.append(params)
        params.requires_grad_(True)

print(f"Number of parameters to train: {training_count:,}")

# Create an optimizer for the parameters that are
# being trained. (You can pick your optimizer)
optimizer = torch.optim.Adam(training_parameters, lr=0.001)

# Use a cross entropy loss function
loss_function = nn.CrossEntropyLoss()

# Create a file to gather the statistics
stats_file = open("stats.txt", "w")
print("epoch,mean_loss,training_accuracy,validation_accuracy", file=stats_file)

# Start learning loop
start_learning = perf_counter()
for current_epoch in range(NUM_EPOCHS):
    # Let the model know it is being trained so dropout is enabled
    ## Your code here
    model.train()
    # Initialize variables for stats
    total_loss = 0.0
    total_iterations = 0
    total_correct = 0
    total_tests = 0

    # Step through the batches of training data
    for inputs, labels in training_dataloader:
        # Move inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Eable the gradients
        with torch.set_grad_enabled(True):

            # Do the forward pass
            outputs =model(inputs)## Your code here

            # Convert labels to one-hot encoding (on the device)
            # It should have dtype torch.float32
            gtpreds = labels## Your code here

            # Compute the loss
            loss = loss_function(outputs, gtpreds)

            # Get hard predictions from the model
            preds = torch.argmax(outputs, dim=1)

            # Gather data for trainng accuracy calculation
            total_correct += torch.sum(preds == labels).item() ## Your data here
            total_tests += len(labels)

            # Gather data for mean loss calculation
            total_loss += loss.item()
            total_iterations += 1

            # Do backpropagation
            ## Your code here
            loss.backward()
            

            # Update the weights
            ## Your code here
            optimizer.step()
            if total_iterations % 10 == 0:
                print(f"Batch {total_iterations} Loss: {loss.item():.4f}", flush=True)

    # Compute stats
    mean_loss = total_loss / total_iterations
    training_accuracy = total_correct / total_tests

    # Validation
    model.eval()
    total_correct = 0
    total_tests = 0
    with torch.no_grad():
        # Step through batches of validation data
        for inputs, labels in validation_dataloader:

            # Move the inputs and labels onto the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Do inference
            outputs = model(inputs)## Your code herer

            # Convert outputs to a 1-d tensor of hard predictions
            preds = torch.argmax(outputs, dim=1)

            # Gather accuracy stats
            total_correct += torch.sum(preds== labels).item()
            total_tests += inputs.size(0)

    # Compute validation accuracy
    validation_accuracy = total_correct / total_tests

    # Store stats
    print(f"{current_epoch + 1},{mean_loss:.6f},{training_accuracy:.6f},{validation_accuracy:.6f}", file=stats_file)

    # Show stats
    print(f'Epoch {current_epoch + 1:<3}:\n\tLoss: {mean_loss:.7f}\n\tTraining Accuracy: {training_accuracy*100.0:.1f}%\n\tValidation Accuracy: {validation_accuracy*100.0:.1f}%', flush=True)


learning_duration = perf_counter() - start_learning

print(f"Time elapsed: {learning_duration:.2f} seconds")

stats_file.close()

# Save model checkpoint
torch.save(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
