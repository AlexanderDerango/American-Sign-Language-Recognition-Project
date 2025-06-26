# Machine Learning Student Network Spring 2025 Project

## American Sign Language Hand Recognition

This project implements a real-time American Sign Language (ASL) hand sign recognition system for the letters A through E using deep learning and computer vision. It uses a pre-trained CNN model to classify ASL hand gestures from a webcam feed, powered by TensorFlow, OpenCV, and MediaPipe. The user will be able to hold up any letter A-E, and the program will be able to tell the user which letter it is.

### Project Structure
* ASL_model.h5 – Trained TensorFlow model for classifying ASL letters A–E.
* asl_demo.py – Live webcam-based ASL detection and classification script.
* asl_dataset/ – Dataset folder with training images of ASL hand signs.
* training_script.py – Script used to train and validate the CNN model.
* requirements.txt – Python dependencies.

### Model Training
* Dataset: ASL Alphabet dataset (A–E subset).
* Image Size: 224x224 for MobileNet compatibility.
* Preprocessing:
  * Rescaling and normalization.
  * Data augmentation (rotation, zoom, shifts, horizontal flip).
* Model: Transfer learning using MobileNetV2 + custom dense layers.
* Evaluation: Achieved high accuracy on validation data.

### Instructions for Running the Live Demo
Requirements:
* pip install -r requirements.txt
* Need to be running Python 3.10 or 3.11
