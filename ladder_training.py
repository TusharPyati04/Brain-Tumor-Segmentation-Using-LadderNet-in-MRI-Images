import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from LadderNetv65 import LadderNetv6 

# Evaluate accuracy
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100  # Return accuracy as percentage

# Train the model with a progress bar
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        train_accuracy = evaluate_accuracy(model, train_loader, device)
        test_accuracy = evaluate_accuracy(model, test_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    print("Training complete.")

# Load preprocessed datasets
train_dataset = datasets.ImageFolder(
    root="D:/COLLEGE/MLDL/Project/codes2/Preprocessed_Training",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

test_dataset = datasets.ImageFolder(
    root="D:/COLLEGE/MLDL/Project/codes2/Preprocessed_Testing",  
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, and optimizer
num_classes = 4
model = LadderNetv6(layers=3, filters=16, num_classes=num_classes, inplanes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, device=device)

# Save the model
model_path = "best_laddernet_model_v6.pth"
torch.save(model.state_dict(), model_path)
print("Model saved at", model_path)

# Load the model
loaded_model = LadderNetv6(layers=3, filters=16, num_classes=num_classes, inplanes=3)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
print("Model loaded successfully.")