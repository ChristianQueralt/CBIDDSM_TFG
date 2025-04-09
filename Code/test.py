import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from resnet_model import BreastCancerResNet18
from dataset import ResNetCBISDDSM
from transforms import test_transforms
from config import TEST_CSV, IMG_DIR, BATCH_SIZE, DEVICE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
import time
import os

def evaluate_model(model, save_dir, train_time):
    """
    Evaluates the trained ResNet18 model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        save_dir (str): Directory where results will be saved.
        train_time (float): Total training time in seconds.
    """
    print("\nStarting Evaluation Process...")

    # Load test dataset
    test_dataset = ResNetCBISDDSM(csv_file=TEST_CSV, root_dir=IMG_DIR, transform=test_transforms)

    # Create DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Set model to evaluation mode
    model.to(DEVICE)
    model.eval()

    # Lists to store predictions and true labels
    y_true, y_pred, y_probas = [], [], []

    start_time = time.time()

    # Evaluate on the test set
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

            # Get model predictions
            outputs = model(images)

            # Apply Softmax for probability scores
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # Store true labels and predictions
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions)
            y_probas.extend(probabilities)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probas = np.array(y_probas)  # Shape should be (num_samples, num_classes)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probas, multi_class='ovr')

    test_time = time.time() - start_time

    # Save results
    results_path = os.path.join(save_dir, "accuracy_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Training Time: {train_time:.2f} sec\n")
        f.write(f"Testing Time: {test_time:.2f} sec\n")

    print("\nEvaluation results saved!")

    # 🔹 Plot ROC Curve
    num_classes = y_probas.shape[1]  # Get number of classes
    plt.figure(figsize=(8, 6))

    class_names = [
        "Malignant Calcification",
        "Malignant Mass",
        "Benign Calcification",
        "Benign Mass"
    ]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probas[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.2f})')


    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    # Labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save ROC Curve
    roc_curve_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()

    print(f"\nROC Curve saved at: {roc_curve_path}")

    # 🔹 Plot Confusion Matrix
    class_names = [
        "Malignant Calc",
        "Malignant Mass",
        "Benign Calc",
        "Benign Mass"
    ]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)

    # Save Confusion Matrix
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()

    print(f"\nConfusion Matrix saved at: {conf_matrix_path}")
