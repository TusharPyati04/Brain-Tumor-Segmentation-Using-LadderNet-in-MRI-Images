# Brain Tumor Segmentation Using LadderNet in MRI Images

A deep learning-based segmentation model that leverages the LadderNet architecture to identify and classify brain tumors from MRI scans.

## Overview

This project applies the LadderNet model for accurate brain tumor segmentation using MRI images. The model addresses common challenges in medical imaging like precise tumor localization, limited dataset sizes, and the need for multi-path information flow.

## Motivation

Accurate tumor segmentation is vital for diagnosis and treatment planning. Traditional models struggle with segmentation precision and capturing multiscale contextual information. LadderNet, with its multi-U-Net structure and extensive skip connections, offers a robust solution.

---

##  Dataset

- **Source:** Kaggle Brain MRI Dataset  
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor  
- **Size:** 5712 training images, 1311 testing images  
- **Format:** Preprocessed and resized to 128×128 pixels, with data augmentation

---

##  Architecture - LadderNet

- Derived from U-Net with dual encoder-decoder paths  
- Shared-weight residual blocks for efficient training  
- Extensive skip connections for enhanced feature reuse  
- Final layer uses 1×1 convolution and global average pooling  

---

## Preprocessing

- Resize images to 128x128  
- Apply data augmentation: random flips and rotations  
- Normalize pixel values between 0 and 1  
- Convert to PyTorch tensors  

---

##  Training Details

- **Model:** Custom `LadderNetv6`  
- **Layers:** 3  
- **Filters:** 16  
- **Learning Rate:** 0.001  
- **Output Classes:** 4  
- **Environment:**  
  - CPU: AMD Ryzen 7 7735HS  
  - GPU: NVIDIA RTX 4050  
  - RAM: 16GB DDR5  

---

##  Evaluation Metrics

- **Classification:** Accuracy, Confusion Matrix  
- **Segmentation:** IoU, Pixel Accuracy, FWIoU  
- **Discrimination:** ROC AUC, Precision-Recall Curves  

---

##  Results

| Metric              | Score       |
|---------------------|-------------|
| Mean Accuracy       | 90.82%      |
| ROC AUC (Overall)   | 95.04%      |

📌 *Note: Visual outputs include confusion matrix, ROC, and precision-recall plots.*

---

##  How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/laddernet-brain-tumor-segmentation.git
cd laddernet-brain-tumor-segmentation

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run predictions
python predict.py

---

