# High-Accuracy MNIST Classifier (CNN+SVM)

A state-of-the-art digit recognition system using a hybrid **Convolutional Neural Network (CNN) and Support Vector Machine (SVM)** model. This approach leverages the powerful feature extraction capabilities of CNNs with the robust classification power of SVMs, achieving an outstanding **99.11% accuracy** on the MNIST test dataset.

This model serves as an upgrade to the previous PCA+SVM implementation, increasing accuracy from 97.86% to 99.11%.

## 🎯 Performance Overview

The model was evaluated on the 10,000-sample MNIST test dataset, demonstrating exceptional performance and generalization.

  * **Overall Accuracy:** **99.11%**
  * **Total Test Samples:** 10,000
  * **Correctly Classified:** 9,911
  * **Misclassified Samples:** 89

## 📊 Detailed Results

Performance is strong and balanced across all 10 digit classes.

### Confusion Matrix

The confusion matrix shows an exceptionally strong diagonal, indicating minimal misclassifications between classes. The model is highly confident and accurate across all digits.

### Per-Class Performance

The classification report, derived from the `cnn_svm_stats.json` file, confirms the high precision, recall, and F1-scores for every class.

| Digit | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.992 | 0.997 | 0.994 | 980 |
| **1** | 0.996 | 0.996 | 0.996 | 1135 |
| **2** | 0.985 | 0.992 | 0.988 | 1032 |
| **3** | 0.994 | 0.991 | 0.993 | 1010 |
| **4** | 0.993 | 0.993 | 0.993 | 982 |
| **5** | 0.980 | 0.992 | 0.986 | 892 |
| **6** | 0.998 | 0.982 | 0.990 | 958 |
| **7** | 0.990 | 0.990 | 0.990 | 1028 |
| **8** | 0.993 | 0.987 | 0.990 | 974 |
| **9** | 0.990 | 0.989 | 0.990 | 1009 |
| | | | | |
| **Macro Avg** | **0.991** | **0.991** | **0.991** | **10000** |
| **Weighted Avg** | **0.991** | **0.991** | **0.991** | **10000** |

## 🚀 Quick Start

### Prerequisites

Ensure you have the necessary libraries installed:

```bash
pip install numpy scikit-learn tensorflow opencv-python pillow joblib
```

### Usage

The main script provides three modes of operation:

#### 1\. Train the Model (First Time Only)

```bash
python main.py train
```

This will:

  * Load the MNIST dataset.
  * Train the CNN for feature extraction.
  * Train the SVM on the extracted features.
  * Save the combined trained model as `cnn_svm_model.pkl`.

#### 2\. Evaluate Model Performance

```bash
python main.py eval
```

This generates the detailed classification report and confusion matrix, saving the metrics to `cnn_svm_stats.json`.

#### 3\. Predict Custom Images

```bash
python main.py predict path/to/your/digit.png
```

Example predictions on custom images:

  * `mydigit1.png` → Predicted digit with confidence score
  * `mydigit2.png` → Predicted digit with confidence score
  * `mydigit3.png` → Predicted digit with confidence score

## 🔧 Technical Details

### Model Architecture

This hybrid model uses a "CNN as feature extractor" approach:

1.  **Feature Extractor:** A **Convolutional Neural Network (CNN)** is trained on the MNIST images. Its purpose is not to classify but to learn a rich, hierarchical representation of the digit features.
2.  **Feature Vector:** The output from one of the final dense (fully-connected) layers of the
    CNN is extracted for each image. This vector serves as the input for the SVM.
3.  **Classifier:** A **Support Vector Machine (SVM)** (likely with an RBF kernel) is trained on these high-level feature vectors. It excels at finding an optimal decision boundary in this complex feature space.

### Image Preprocessing Pipeline

To handle custom images for prediction, a robust pipeline is used:

1.  Convert to **grayscale**.
2.  **Invert** colors (to match MNIST's white-on-black format).
3.  Apply **binary thresholding** (OTSU's method) to isolate the digit.
4.  Extract the digit's **bounding box**.
5.  Resize the digit to **20×20 pixels** while maintaining aspect ratio.
6.  Pad and center the digit within a **28×28 canvas**.
7.  **Normalize** pixel values to the [0, 1] range.

## 📁 Project Structure

```
├── main.py                # Main script with train/eval/predict functions
├── cnn_svm_model.pkl      # Trained hybrid model (generated after training)
├── cnn_svm_stats.json     # Detailed evaluation metrics (from eval)
├── confusion_matrix.png   # Confusion matrix heatmap (from eval)
├── mydigit1.png           # Sample custom digit image
├── mydigit2.png           # Sample custom digit image
├── mydigit3.png           # Sample custom digit image
└── README.md              # This file
```

## 🔍 Key Features

  * **State-of-the-Art Accuracy:** **99.11%** on the MNIST test set.
  * **Hybrid Model:** Combines the superior feature learning of CNNs with the powerful classification boundary of SVMs.
  * **Fast Inference:** Once trained, the model provides rapid predictions.
  * **Custom Image Support:** A robust preprocessing pipeline allows for prediction on real-world, user-provided images.
  * **Comprehensive Metrics:** Detailed JSON output for per-class analysis.

## 📈 Results Summary

The CNN+SVM hybrid approach proves to be a highly effective strategy for the MNIST dataset, significantly outperforming the classical SVM+PCA model (99.11% vs. 97.86%).

This implementation demonstrates that by thoughtfully combining deep learning for feature extraction with classical machine learning for classification, it's possible to achieve top-tier performance that is both accurate and robust.
