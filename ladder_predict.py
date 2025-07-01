import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve,
    f1_score, jaccard_score
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from LadderNetv65 import LadderNetv6 

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = datasets.ImageFolder(
    root="Preprocessed_Testing",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load trained model
num_classes = 4
model = LadderNetv6(layers=3, filters=16, num_classes=num_classes, inplanes=3).to(device)
model.load_state_dict(torch.load("best_laddernet_model_v6.pth"))
model.eval()

# Initialize lists for storing results
y_true, y_pred, y_probs = [], [], []

# Perform predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Store results
        y_true.extend(labels.cpu().numpy())
        y_probs.extend(outputs.cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())

# Convert lists to numpy arrays
y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Compute additional metrics
roc_auc = roc_auc_score(np.eye(num_classes)[y_true], y_probs, multi_class="ovr")
f1 = f1_score(y_true, y_pred, average="weighted")
jaccard = jaccard_score(y_true, y_pred, average="weighted")
pixel_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

# Print results
print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"Overall ROC AUC: {roc_auc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"Jaccard Score (weighted): {jaccard:.4f}")
print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
