import time
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from resnet_model import BreastCancerResNet18
from dataset import ResNetCBISDDSM
from transforms import train_transforms, test_transforms
from config import TRAIN_CSV, VAL_CSV, IMG_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE
import os
import datetime
import matplotlib.pyplot as plt
from test import evaluate_model

# Early stopping parameters
PATIENCE = 15  # Number of epochs to wait before stopping if no improvement

def train_model():
    """
    Function to train the ResNet18 model for breast cancer classification.
    Automatically calls evaluate_model() after training.
    """
    # Load datasets
    train_dataset = ResNetCBISDDSM(csv_file=TRAIN_CSV, root_dir=IMG_DIR, transform=train_transforms)
    val_dataset = ResNetCBISDDSM(csv_file=VAL_CSV, root_dir=IMG_DIR, transform=test_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_images = next(iter(train_loader))["image"]
    for ig in train_images:
        print(ig.min(), ig.max(), ig.shape)
    # Usamos make_grid para organizar las imágenes en una cuadrícula
    grid_img = torchvision.utils.make_grid(train_images, nrow=4, normalize=True)

    # Convertimos el tensor a NumPy y transponemos para el formato correcto (H, W, C)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convertimos de (C, H, W) -> (H, W, C)
    plt.axis("off")  # Ocultamos los ejes
    plt.savefig(r"/home/christian/Resoults/TEST/Last_Test.png")


    # Initialize the model
    model = BreastCancerResNet18(pretrained=True).to(DEVICE)

    # Define the loss function (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (Adam with learning rate)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop variables
    train_loss_history = []
    val_loss_history = []

    # Early stopping variables
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    patience_counter = 0  # Counter to track epochs without improvement

    # Start timing the training process
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Switch model to training mode
        model.train()
        running_loss = 0.0

        # Iterate over training batches
        for batch in train_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch}/{NUM_EPOCHS} - Training Loss: {epoch_loss:.4f} - Validation Loss: {val_loss:.4f} - Time: {epoch_duration:.2f} sec")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter when validation loss improves
        else:
            patience_counter += 1  # Increment counter if no improvement

        # Stop training if patience limit is reached
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # End timing
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print("\nTotal Training Time:", total_training_time / 3600, "hours")

    # Save results after training
    trial_dir = save_results(train_loss_history, val_loss_history, total_training_time)

    # 🔹 Call evaluate_model() automatically after training and pass the save directory
    evaluate_model(model, trial_dir, total_training_time)


def save_results(train_loss_history, val_loss_history, train_time):
    """
    Saves the training results in a timestamped folder inside /home/christian/Resoults.

    Args:
        train_loss_history (list): List of training loss values per epoch.
        val_loss_history (list): List of validation loss values per epoch.
        train_time (float): Total training time in seconds.

    Returns:
        str: Path of the directory where results were saved.
    """
    # Create the main results directory if it doesn't exist
    results_dir = "/home/christian/Resoults"
    os.makedirs(results_dir, exist_ok=True)

    # Create a folder with the current date and time
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    trial_dir = os.path.join(results_dir, timestamp)
    os.makedirs(trial_dir, exist_ok=True)

    # Save training loss graph as a .png file
    graph_path = os.path.join(trial_dir, "graph.png")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Training Loss")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(graph_path)
    plt.close()

    # Save settings in a text file
    settings_path = os.path.join(trial_dir, "settings.txt")
    with open(settings_path, "w") as f:
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"PATIENCE: {PATIENCE}\n")  # Save patience parameter

    print(f"Results saved in: {trial_dir}")

    return trial_dir


if __name__ == "__main__":

    train_model()  # Train model and automatically evaluate it
