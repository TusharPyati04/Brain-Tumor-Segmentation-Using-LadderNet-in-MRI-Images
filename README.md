# Brain Tumor Segmentation Using LadderNet in MRI Images

A deep learning-based segmentation model that leverages the LadderNet architecture to identify and classify brain tumors from MRI scans.

## Overview

This project applies the LadderNet model for accurate brain tumor segmentation in MRI images. LadderNet builds upon the U-Net architecture by introducing multiple encoder-decoder paths and extensive skip connections, enabling better feature extraction and improved segmentation accuracy. It effectively handles challenges like limited medical data, poor tumor localization, and variability in tumor appearance. Trained on a labeled Kaggle dataset, the model achieves strong results with a mean accuracy of 90.82% and a ROC AUC of 95.04%, making it suitable for real-world medical image analysis.


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
## Contributors

- **Vineet Desai** 
- **Tushar Pyati** 
- **K L Bhargava Prasad** 

 
**School of Electronics and Communication Engineering**  
**KLE Technological University, Hubballi**

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

