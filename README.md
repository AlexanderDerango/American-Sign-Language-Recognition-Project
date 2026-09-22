# American Sign Language Hand Recognition with MobileNetV2

## Project Overview

This project was developed as part of the **Machine Learning Student Network (MLSN) Spring 2025 Project**.

The goal was to build a machine learning system capable of recognizing American Sign Language hand gestures in real time. The project combines image classification, transfer learning, computer vision, and webcam-based hand detection.

The system recognizes five ASL letters:

* A
* B
* C
* D
* E

A user can hold up one of these hand signs in front of a webcam, and the system detects the hand region and predicts the corresponding letter.

## Project Pipeline

The project follows a complete machine learning pipeline:

**Dataset → Preprocessing → Data Augmentation → Model Training → Model Evaluation → Webcam Detection → Real-Time Prediction**

### Dataset

The project uses the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle.

For this project, the dataset was limited to the first five ASL letters:

* **A:** 500 images
* **B:** 500 images
* **C:** 500 images
* **D:** 500 images
* **E:** 500 images

**Total: 2,500 images**

The dataset was split into:

* **80% training data**
* **20% testing/validation data**

## Data Preprocessing

The images were prepared for use with MobileNetV2.

### Preprocessing Steps

1. Organized the images into class-specific folders.
2. Resized images to **224 × 224 pixels**.
3. Normalized pixel values.
4. Applied data augmentation to the training images.

### Data Augmentation

Data augmentation was used to create variations of the training images and help the model generalize to different hand positions.

Augmentation included:

* Rotation
* Zoom
* Horizontal shifting
* Vertical shifting
* Horizontal flipping

This helped expose the model to variations that may occur when using the webcam.

## Model Architecture

The project uses **MobileNetV2** through transfer learning.

MobileNetV2 was selected as a pre-trained convolutional neural network that can provide useful visual features while keeping the model relatively lightweight for real-time applications.

The base MobileNetV2 layers were frozen and custom classification layers were added for the five ASL classes.

### Architecture

```text
MobileNetV2
     ↓
GlobalAveragePooling2D
     ↓
Dense(128, ReLU)
     ↓
Dropout(0.3)
     ↓
Dropout(0.2)
     ↓
Dense(5, Softmax)
```

The final softmax layer produces predictions for:

```text
A, B, C, D, E
```

### Model Components

* **MobileNetV2:** Extracts visual features from the hand images.
* **GlobalAveragePooling2D:** Reduces the spatial dimensions of the extracted features.
* **Dense(128, ReLU):** Learns features specific to the ASL classification task.
* **Dropout:** Helps reduce overfitting.
* **Dense(5, Softmax):** Produces probabilities for the five ASL letters.

### Model Training Configuration

| Setting       | Value                            |
| ------------- | -------------------------------- |
| Model         | MobileNetV2                      |
| Image Size    | 224 × 224                        |
| Optimizer     | Adam                             |
| Loss Function | Sparse Categorical Cross-Entropy |
| Metric        | Accuracy                         |
| Epochs        | 15                               |
| Classes       | 5                                |
| Classes       | A, B, C, D, E                    |

## Results

The model showed steady improvement in both training and validation accuracy over the 15 training epochs.

Final reported performance:

* **Training Accuracy:** ~91%
* **Validation Accuracy:** ~94%

The validation accuracy indicates that the model was able to generalize well to images that were not used for training.

## Real-Time Webcam Prediction

After training, the model was saved and loaded into a local Python environment for real-time webcam testing.

The live prediction pipeline uses:

* **OpenCV** for webcam input and video processing
* **MediaPipe** for hand detection
* **TensorFlow/Keras** for ASL classification
* **NumPy** for image preprocessing

### Real-Time Pipeline

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
Hand Region Extraction
   ↓
Resize to 224 × 224
   ↓
Normalize Pixel Values
   ↓
MobileNetV2 Model
   ↓
ASL Letter Prediction
   ↓
Display Prediction on Webcam
```

The hand region is resized and normalized before being passed to the trained model. Then, the predicted ASL letter is then displayed directly on the webcam feed.

## Project Structure

```text
.
├── MLSN_project.ipynb
├── asl_demo.py
├── ASL_model.h5
├── requirements.txt
└── README.md
```

### Files

**`MLSN_project.ipynb`**
Jupyter/Google Colab notebook containing dataset preparation, preprocessing, model training, and evaluation.

**`asl_demo.py`**
Python script for real-time webcam-based ASL hand detection and classification.

**`ASL_model.h5`**
Saved TensorFlow/Keras model trained to classify ASL letters A–E.

**`requirements.txt`**
Python dependencies required to run the project.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

Python **3.10 or 3.11** is recommended because of compatibility between TensorFlow, MediaPipe, and other dependencies.

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Live Demo

After installing the dependencies and ensuring that the trained model is available:

```bash
python asl_demo.py
```

The program will open the webcam and begin detecting hand gestures.

Hold up an ASL letter from **A–E** in front of the camera. The model will display its predicted letter on the video feed.

### Recommended Setup

For the live demonstration:

* Use a working webcam.
* Make sure your hand is clearly visible.
* Use adequate lighting.
* Position your hand within the camera frame.
* Hold one ASL gesture at a time.

## Development Environment

The project was developed using a combination of Google Colab and Visual Studio Code.

### Google Colab

Google Colab was used for:

* Dataset preparation
* Model training
* Model evaluation
* Experimentation with the TensorFlow model

### Visual Studio Code

VS Code was used for:

* Local development
* Webcam testing
* OpenCV integration
* MediaPipe integration
* Real-time model inference

## Challenges and Lessons Learned

### 1. Dataset Size and Scope

The original goal was to build a larger ASL recognition system. Due to computational limitations when working with the free version of Google Colab, the project was initially limited to five letters.

This reduced the problem to a manageable classification task while allowing the team to focus on building the complete pipeline from training to real-time prediction.

### 2. Dependency Compatibility

The project encountered compatibility issues between different versions of:

* TensorFlow
* Keras
* NumPy
* MediaPipe

Resolving these issues required testing different package versions and creating a compatible Python environment.

### 3. Moving from Colab to Local Development

Google Colab was useful for training the model, but real-time webcam testing required a local environment.

The project therefore transitioned from Google Colab to VS Code for the webcam implementation using OpenCV and MediaPipe.

### 4. Improving Model Predictions

During early webcam testing, the model had difficulty distinguishing certain gestures.

In particular:

* B and D were not consistently detected.
* A and E were sometimes confused.

This required additional model training and experimentation with preprocessing and the training pipeline.

## What I Learned

Through this project, I gained hands-on experience with:

* Image classification
* Transfer learning
* Convolutional neural networks
* TensorFlow and Keras
* MobileNetV2
* Data augmentation
* Image preprocessing
* OpenCV
* MediaPipe
* Real-time computer vision
* Model evaluation
* Python environment management
* Debugging machine learning dependencies
* Team-based machine learning development

The project also provided experience taking a model beyond training and integrating it into a real-time application.

## Future Work

Potential improvements include:

### Expand ASL Recognition

* Expand the classifier from 5 letters to all 26 ASL letters.
* Increase the size and diversity of the training dataset.
* Explore recognition of simple ASL words and phrases.

### Improve Model Performance

* Fine-tune MobileNetV2 instead of using only frozen base layers.
* Experiment with different architectures.
* Add more varied training images.
* Improve hand-region preprocessing.
* Continue addressing confusion between visually similar gestures.

### Real-Time Improvements

* Improve prediction stability across consecutive frames.
* Add confidence thresholds for predictions.
* Add temporal smoothing to reduce prediction changes caused by individual frames.
* Develop a more polished real-time user interface.

## Technologies

* **Python**
* **TensorFlow**
* **Keras**
* **MobileNetV2**
* **OpenCV**
* **MediaPipe**
* **NumPy**
* **Google Colab**
* **Visual Studio Code**
* **Git/GitHub**

## Acknowledgments

This project was developed as part of the **Machine Learning Student Network (MLSN) Spring 2025 Project**.

The ASL image data was obtained from the Kaggle **ASL Alphabet Dataset** by GrassKnoted.
