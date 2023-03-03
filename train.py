import torch
from torch.optim import Adam
import torch.nn as nn

import time
from tqdm import tqdm

from model import UNet,unet
from data import train_loader,val_loader,train_dataset,val_dataset,batch_size

# Initialise UNet model
model = unet(1)
# Set learning rate
lr = 0.0001
# Set number of epochs
epochs = 2
# Define loss function
loss = nn.BCEWithLogitsLoss()
# Define optimizer
optimizer = Adam(model.parameters(),lr=lr)
# Calculate steps per epoch for training, validation and test set
train_steps = len(train_dataset)//batch_size
val_steps = len(val_dataset)//batch_size
# Initialize a dictionary store training history
history = {"train_loss":[],"val_loss":[]}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Loop over epochs
print("Training Network ... ")
start_time = time.time()
for epoch in tqdm(range(epochs)):
    # Set the model in training mode
    model.train()
    # Initialize the total training and validation loss
    train_loss = 0
    val_loss = 0
    # Loop over the training set
    for (i,(images,masks)) in enumerate(train_loader):
        # Send the input to the device
        (images,masks) = (images.to(device),masks.to(device))
        # Perform a forward pass
        pred = model(images)
        # Calculate the training loss
        loss = loss(pred,masks)
        # Zero out any previously accumulated gradients
        optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Update model parameters
        optimizer.step
        # Add the loss to the total training loss so far
        train_loss += loss

    # Turn off autograd
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        # Loop over the validation set
        for (images,masks) in val_loader:
            # Send the input to the device
            (images,masks) = (images.to(device),masks.to(device))
            # Make the predictions
            pred = model(images)
            # Calculate the validation loss
            val_loss += loss(pred,masks)
    # Calculate the average training and validation loss
    avg_train_loss = train_loss/train_steps
    avg_val_loss = val_loss/val_steps
    # Update training history
    history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    history['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    # Print the model training and validation information
    print(f"Epoch: {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.4f}")
# Display the total time needed for training
end_time = time.time()
print(f"Total training time: {end_time-start_time:.2f}s")

