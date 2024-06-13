# ECE_228_Project
# UAV Safe Landing Site Detection

## Overview
This project is developed as part of the coursework for ECE 228. It aims to detect safe landing sites for Unmanned Aerial Vehicles (UAVs) by analyzing environmental data through image processing techniques. The system utilizes a combination of segmentation, object detection, and optical flow algorithms to evaluate and identify suitable landing zones in real-time.

## Modules
The project consists of several Python modules that work together to process images, detect features, and compute safe landing areas:

### 1. **Segmentation**
- **File**: `segmentation.py`
- **Description**: This module uses a DeepLabV3 model with a ResNet50 backbone to segment the input images into predefined categories, which are crucial for identifying different types of surfaces and obstacles.

### 2. **Object Detection**
- **File**: `obstacle_detection_utils.py`
- **Description**: Implements object detection to identify potential obstacles using a Faster R-CNN model. This helps in recognizing objects that might pose risks for landing, such as people, vehicles, and other significant obstructions.

### 3. **Optical Flow**
- **File**: `optical_flow.py`
- **Description**: Calculates optical flow to determine the motion pattern between sequential frames, which assists in identifying moving obstacles and estimating the flatness of the area based on motion consistency.

### 4. **Utilities**
- **Files**: `utils.py`, `train.py`
- **Description**: Contains utility functions for training models, calculating metrics like IoU and pixel accuracy, and visualizations of results.

### 5. **Training Scripts**
- **Files**: `train.ipynb`, `train.py`
- **Description**: Scripts and notebooks for training the segmentation and object detection models, including evaluation and validation procedures.

To see the final results, run the `inference.ipynb` notebook. This notebook provides a step-by-step visualization of the processing pipeline and the output safe landing zones.

## Setup and Installation
Ensure you have Python 3.8+ installed along with the following packages:
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib

## Datasets

This project utilizes the following datasets:

- **TU-Graz Landing Dataset**: Images for testing UAV landing site detection algorithms. [Access the dataset here](http://dronedataset.icg.tugraz.at).
- **Aeroscapes**: Aerial segmentation dataset providing images for a variety of aerial scenes. [Access the dataset on GitHub](https://github.com/ishann/aeroscapes).

