# Brain Tumor Detection Using CNN
## Project Overview

The advent of deep learning technology has significantly advanced the field of medical imaging, particularly in the detection of brain tumors. Convolutional Neural Networks (CNN), a powerful deep learning algorithm, offer the capability to analyze MRI scans with high precision and efficiency. By automating the detection process, CNN models reduce the time required for manual image interpretation while also minimizing the risk of human error, ultimately leading to better patient outcomes.

This project aims to develop a CNN-based model using the PyTorch framework to accurately detect brain tumors from MRI images. By training on a dataset of labeled brain tumor images, the model will learn to identify specific patterns associated with tumor presence, making it a valuable tool to support healthcare professionals in the diagnosis process.


![brain-tumor-fig1](https://github.com/user-attachments/assets/f36b4bbe-0044-4dce-b04d-95391011d95e)

## Project Aim
The aim of this project is to build a reliable and automated deep-learning model that can detect brain tumors from MRI scans with high accuracy. This tool will assist medical professionals by providing faster, more accurate diagnoses, reducing human error, and improving overall efficiency in patient care.

## Project Objective
The primary objectives of this initiative are to:

1. Develop and train a Convolutional Neural Network (CNN) using PyTorch to accurately detect brain tumors.

2. Achieve a high level of accuracy in identifying the presence of tumors from MRI images.

3. Provide healthcare professionals with an innovative tool that enhances the speed and accuracy of tumor diagnosis, reducing dependency on manual interpretation and improving overall diagnostic efficiency.

By achieving these objectives, this project will deliver a solution capable of expediting diagnosis and treatment planning, improving patient care and clinical outcomes.

## Dataset and Input
Dataset:
The dataset used in this project consists of MRI images of brain scans, labeled as either tumor-positive or tumor-negative.

**Input Format:**

**Image Size:** Images are typically resized to a fixed size (e.g., 224x224 pixels) for input to the model.

**Channels:** MRI images are usually grayscale (single-channel), but they may be transformed into 3-channel images to match the expected input for CNNs like those used in transfer learning
models (e.g., ResNet).

**Labeling:** The input images are labeled as either Healthy  or Brain Tumor, allowing for binary classification.

Example of an input image: ![ccd26633-b137-47d8-8f4b-ffaef143eb5f](https://github.com/user-attachments/assets/98dee1bc-f27a-402e-b6e5-770e5998c1e4)

## Steps Followed

This section outlines the key steps followed in developing the project:

**Data Collection:**

The dataset was collected from a public source (mention the source, e.g., Kaggle) that provides MRI scans labeled as either tumor or non-tumor.

**Data Preprocessing:**

MRI images were resized to a uniform size.
Normalization and augmentation techniques were applied to increase the diversity of training data (rotation, flipping, etc.).
The dataset was split into training, validation, and test sets.

**Model Development:**

A Convolutional Neural Network (CNN) architecture was built using PyTorch.
Transfer learning techniques (using pre-trained models such as ResNet or VGG) were considered to improve model performance.
The model was trained using cross-entropy loss for binary classification and an Adam optimizer for learning.
Training and Evaluation:

The model was trained over several epochs with periodic validation to monitor accuracy and loss.
Early stopping and learning rate scheduling techniques were implemented to optimize training.

**Model Evaluation:**

The model was evaluated on a separate test dataset, and metrics such as accuracy, confusion matrix, and classification report were generated to assess performance.

## Results

The performance of the model was evaluated using a test dataset, and the following metrics were obtained:

**Confusion Matrix**


The confusion matrix provides a summary of the prediction results, showing the number of correct and incorrect predictions for each class (tumor/no tumor).


True Positives (TP): Correctly identified brain tumors.

True Negatives (TN): Correctly identified cases with no tumors.

False Positives (FP): Incorrectly classified no tumor cases as tumors.

False Negatives (FN): Incorrectly classified tumor cases as no tumors.

This visualization helps identify where the model is making mistakes (if any) and provides insight into its overall performance.

![bf581a7f-a828-41d1-b41b-622e0eaf2824](https://github.com/user-attachments/assets/eb4f6722-d4b3-43fd-b7a8-e7e6861635d5)


**Training Accuracy vs. Epochs**

The graph below demonstrates the model's accuracy over each epoch of training, showing how the performance improved during the learning process.


The plot shows:

X-axis: Epochs (number of training iterations).

Y-axis: Accuracy (performance on the validation dataset).

The accuracy generally improves as the number of epochs increases, which indicates that the model is learning and becoming more proficient at classifying the input images.

![3e74ab04-92cd-4f3d-9202-1479483ed0d8](https://github.com/user-attachments/assets/fcd6f363-2dee-4df6-a2ec-294c5f6b5fff)



