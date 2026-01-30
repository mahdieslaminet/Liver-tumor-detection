# AI-articles

📁 Google Drive Folder:
https://drive.google.com/drive/folders/177Eki7CnZlgADdPXg4D5fvWP0juI7onx

📄 Papers Included

1. Dermatologist-level Classification of Skin Cancer with Deep Neural Networks

2. Medical Image Analysis

3. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs

4. Artificial Intelligence-Supported Screen Reading versus Standard Double Reading in the Mammography Screening with Artificial Intelligence Trial (MASAI): A Clinical Safety Analysis of a Randomised, Controlled, Non-Inferiority, Single-Blinded, Screening Accuracy Study

5. Using Generative AI to Improve the Performance and Interpretability of Rule-Based Diagnosis of Type 2 Diabetes Mellitus

---
# Best Model

Download from here: https://drive.google.com/drive/folders/177Eki7CnZlgADdPXg4D5fvWP0juI7onx


---

# Liver Tumor Segmentation using U-Net

## 📌 Project Overview
This project implements a Deep Learning model to automatically detect and segment liver tumors from CT scan images. Using the **LiTS (Liver Tumor Segmentation Benchmark)** dataset, the system is trained to distinguish between healthy liver tissue/background and tumor lesions.

This tool is designed to assist medical professionals by providing a "second pair of eyes," potentially speeding up diagnosis and treatment planning for liver cancer.

## 🚀 How It Works

The project relies on **Semantic Segmentation**, where the goal is to classify every single pixel in an image as either "Tumor" or "Background."

### 1. The Architecture: U-Net
We use the **U-Net architecture**, a specialized Convolutional Neural Network (CNN) designed for biomedical image segmentation. It consists of two main paths:
* **The Encoder (Contraction Path):** Captures the "context" of the image (what objects are present) by downsampling the image and increasing feature channels.
* **The Decoder (Expansion Path):** Enables precise "localization" (where the objects are) by upsampling the features back to the original image size.
* **Skip Connections:** Critical links that pass high-resolution details from the Encoder directly to the Decoder, ensuring fine details (like tumor edges) are preserved.

### 2. Data Pipeline
* **Source:** LiTS Challenge Dataset (Volume and Segmentation .png slices).
* **Preprocessing:**
    * **Resizing:** All CT scans are resized to `256x256` pixels for efficiency.
    * **Normalization:** Pixel intensity is scaled to a `0-1` range to help the neural network converge faster.
    * **Filtering:** The training dataset is balanced to focus on slices containing tumors, preventing the model from becoming biased toward empty background slices.

### 3. Training Strategy
* **Loss Function:** We use **Dice Loss** instead of standard Cross-Entropy accuracy.
    * *Why?* Tumors are often very small compared to the rest of the body. Standard accuracy would be high even if the model missed the tumor entirely. Dice Loss specifically penalizes the model for the lack of *overlap* between the predicted tumor and the real tumor.
* **Optimization:** The model is trained using the **Adam optimizer** with dynamic learning rate adjustment (`ReduceLROnPlateau`) and `EarlyStopping` to prevent overfitting.

## 🛠️ Tech Stack
* **Language:** Python
* **Frameworks:** TensorFlow / Keras
* **Libraries:** NumPy, OpenCV, Matplotlib, Scikit-learn
* **Environment:** Google Colab / Jupyter Notebook

## 📊 Results
The model outputs a probability map where white pixels represent the predicted location of a tumor. The performance is evaluated using the **Dice Coefficient** (a measure of set similarity).

## 🟢 Try the Live Demo
**Click the badge below to test the AI yourself:**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20Live%20App-blue)](https://huggingface.co/spaces/MobinEs/liver-segmentation-demo)

*(Note: The app runs on a free CPU instance, so prediction might take a few seconds.I've also provided a CT-Scan image in order to test the model. Please download it from the file at the top)*

## 📊 Technical Performance
- **Validation Accuracy (Dice Score):** 93.7%
- **Loss:** 0.0624
- **Model Architecture:** U-Net with Dropout for regularization.

## 📂 Project Structure
- **`Unet_TUMOR.ipynb`**: The training notebook.
- **`app.py`**: The deployment script.
- **`model_best.keras`**: The trained model weights.
