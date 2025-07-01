import os
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import datasets, transforms

# Define a simpler set of transformations for training
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ToTensor(),  # Ensure this always returns a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test transform (no augmentation, only resizing & normalization)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess images and save them
def preprocess_and_save(dataset_path, output_path, transform, class_names):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset = datasets.ImageFolder(root=dataset_path)
    for class_name, class_idx in dataset.class_to_idx.items():
        class_folder = os.path.join(output_path, class_names[class_idx])
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    for img_path, label in dataset.imgs:
        try:
            img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
            img_transformed = transform(img)  # Apply transformations

            # Check if the transformation resulted in a tensor
            if not isinstance(img_transformed, torch.Tensor):
                raise TypeError(f"Error: transform(img) did not return a Tensor. Got {type(img_transformed)} instead.")

            # Convert tensor to NumPy array
            img_transformed = img_transformed.cpu().numpy()  # Convert to NumPy array
            img_transformed = np.transpose(img_transformed, (1, 2, 0))  # Convert CHW -> HWC

            # Denormalize correctly
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_transformed = (img_transformed * std) + mean
            img_transformed = np.clip(img_transformed * 255, 0, 255).astype(np.uint8)

            # Convert back to PIL image
            img_transformed = Image.fromarray(img_transformed)

            # Save image to preprocessed folder
            class_name = class_names[label]
            img_name = os.path.basename(img_path)
            output_file = os.path.join(output_path, class_name, img_name)

            img_transformed.save(output_file)

        except UnidentifiedImageError:
            print(f"Skipping file (not a valid image): {img_path}")
        except TypeError as e:
            print(f"TypeError processing file {img_path}: {e}")
        except Exception as e:
            print(f"Error processing file {img_path}: {e}")

# Class names for glioma, meningioma, no tumor, pituitary
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Paths for input and output
train_input_path = r"./DRIVE/Training"
train_output_path = r"D:/COLLEGE/MLDL/Project/codes2/Preprocessed_Training"
test_input_path = r"./DRIVE/Testing"
test_output_path = r"D:/COLLEGE/MLDL/Project/codes2/Preprocessed_Testing"

# Preprocess and save training and testing datasets
preprocess_and_save(train_input_path, train_output_path, train_transform, class_names)
preprocess_and_save(test_input_path, test_output_path, test_transform, class_names)

print("Preprocessing complete. Preprocessed datasets saved.")
