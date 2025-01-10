# Detecting-Forest-fire Convolutional Neural Network

This project is an attempt to use convolutional neural networks (CNN) to detect the presence or the start of a forest fire in an image. The idea is that this model could be applied to detect a fire or a start of a fire from (aerial) surveillance footage of a forest. The model could be applied in real-time to low-framerate surveillance video (with fires not moving very fast, this assumption is somewhat sound) and give alert in case of fire.

CNN Model (cnn_model.py) – The architecture of the Convolutional Neural Network.
Training Script (train_model.py) – Script to train the CNN model.
Prediction Script (predict_fire.py) – A script to make predictions using the trained model.
Required files for training and prediction handling.


A description of the project, along with examples of video annotation by our network is provided below.

![video_0 (1)](https://github.com/akshaybannatti/Detecting-Forest-Fire-Convolutional-Neural-Network/assets/50884750/378adb83-cc18-49d2-b365-2f14f2626ec2)  

Dataset Overview
The model was trained using a dataset containing images from three categories: 'fire', 'no fire', and 'start fire'. In total, the dataset comprises around 6,000 images, primarily featuring forest or forest-like environments.

'Fire' images contain visible flames.
'Start fire' images feature smoke, indicating the beginning of a fire.
'No fire' images depict forest scenes without any fire or smoke.
Data Augmentation
Through experimentation, we observed that the model struggled to classify 'start fire' images effectively. To address this, we enriched the dataset by extracting frames from videos that depict the early stages of a fire. To improve the model’s ability to generalize, we applied data augmentation techniques provided by Keras. This involved performing random transformations on the images, including zooms, shifts, crops, and rotations, to diversify the training data before it was fed into the model.

Share


You said:
develop a entire project with entire code for Detecting-Forest-Fire-Convolutional-Neural-Network. Write and provide Forest Fire prediction using Machine Learning.txt, naive_approach.py, setup_datasets.py 
ChatGPT said:
ChatGPT
To create a complete project for detecting forest fires using Convolutional Neural Networks (CNNs), I'll outline the structure and provide code for each of the components you requested. This includes:

Forest Fire Prediction using Machine Learning (Forest_Fire_Prediction_using_Machine_Learning.txt).
Naive Approach for Forest Fire Detection (naive_approach.py).
Dataset Setup (setup_datasets.py).
The steps involved will be:

Setting up and preparing the dataset.
Preprocessing the images.
Building the CNN model.
Training and evaluating the model.
Saving and loading the model for predictions.
1. Forest_Fire_Prediction_using_Machine_Learning.txt
This text file will describe the overall machine learning approach used for forest fire detection.

Forest Fire Prediction Using Machine Learning

This project utilizes Convolutional Neural Networks (CNNs) for detecting forest fires in images. The goal is to predict whether an image shows a forest fire or not, based on the analysis of the image's features.

Steps:

Data Collection:

We use an image dataset consisting of two classes: "fire" and "no-fire". The images are captured in various conditions to enhance the model's robustness.
Data Preprocessing:

Images are resized to a uniform shape.
Data augmentation techniques are applied to increase diversity in the dataset, such as random rotations and flips.
Images are normalized to ensure consistency in pixel value ranges.
Model Architecture:

A Convolutional Neural Network (CNN) is used to detect the patterns of fire in images.
The architecture consists of several convolutional layers followed by pooling layers, with a dense fully connected layer at the end for classification.
Model Training:

The dataset is divided into training and testing sets.
The model is trained using the training data, and the performance is evaluated on the testing set.
Performance Evaluation:

Accuracy and other performance metrics like precision, recall, and F1-score are used to evaluate the model's performance.
Final Model:

Once trained, the model can be saved and used for real-time forest fire detection from images.
2. naive_approach.py
This script uses a simple approach (e.g., using basic image processing techniques or a shallow model) to serve as a baseline for forest fire detection.

python
Copy code
import cv2
import numpy as np
import os

# Function to load dataset
def load_data(data_dir):
    fire_images = []
    no_fire_images = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        if folder == 'fire':
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename))
                fire_images.append(cv2.resize(img, (224, 224)))
        elif folder == 'no_fire':
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename))
                no_fire_images.append(cv2.resize(img, (224, 224)))
    
    return np.array(fire_images), np.array(no_fire_images)

# Function to preprocess image data (simple thresholding method)
def preprocess_images(images):
    processed_images = []
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        processed_images.append(thresh)
    
    return np.array(processed_images)

# Example Naive approach: Count the number of "fire-like" pixels in the image
def naive_fire_detection(image):
    # Count number of "fire-like" pixels (based on color threshold)
    fire_pixels = np.sum(image > 200)
    if fire_pixels > 1000:
        return 1  # Fire detected
    return 0  # No fire

if __name__ == "__main__":
    fire_images, no_fire_images = load_data('dataset/')

    # Preprocess images
    fire_images = preprocess_images(fire_images)
    no_fire_images = preprocess_images(no_fire_images)

    # Combine both classes
    images = np.concatenate((fire_images, no_fire_images), axis=0)
    labels = np.concatenate((np.ones(len(fire_images)), np.zeros(len(no_fire_images))), axis=0)

    # Apply the naive approach
    correct_predictions = 0
    for i, img in enumerate(images):
        prediction = naive_fire_detection(img)
        if prediction == labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(labels)
    print(f"Naive Approach Accuracy: {accuracy * 100:.2f}%")
3. setup_datasets.py
This script handles the setup of the dataset, such as downloading, organizing, and splitting the dataset into training and testing data.

python
Copy code
import os
import zipfile
import random
from sklearn.model_selection import train_test_split
import shutil

# Setup directories for the dataset
def setup_directories(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(f'{base_dir}/train/fire'):
        os.makedirs(f'{base_dir}/train/fire')
    if not os.path.exists(f'{base_dir}/train/no_fire'):
        os.makedirs(f'{base_dir}/train/no_fire')
    if not os.path.exists(f'{base_dir}/test/fire'):
        os.makedirs(f'{base_dir}/test/fire')
    if not os.path.exists(f'{base_dir}/test/no_fire'):
        os.makedirs(f'{base_dir}/test/no_fire')

# Unzip dataset if it is in a compressed format
def unzip_data(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Split dataset into train and test sets
def split_dataset(data_dir, train_size=0.8):
    fire_images = os.listdir(f'{data_dir}/fire')
    no_fire_images = os.listdir(f'{data_dir}/no_fire')

    fire_train, fire_test = train_test_split(fire_images, train_size=train_size)
    no_fire_train, no_fire_test = train_test_split(no_fire_images, train_size=train_size)

    # Move files to train and test directories
    for img in fire_train:
        shutil.move(f'{data_dir}/fire/{img}', f'{data_dir}/train/fire/{img}')
    for img in fire_test:
        shutil.move(f'{data_dir}/fire/{img}', f'{data_dir}/test/fire/{img}')
    for img in no_fire_train:
        shutil.move(f'{data_dir}/no_fire/{img}', f'{data_dir}/train/no_fire/{img}')
    for img in no_fire_test:
        shutil.move(f'{data_dir}/no_fire/{img}', f'{data_dir}/test/no_fire/{img}')

if __name__ == "__main__":
    data_dir = 'dataset'
    setup_directories(data_dir)

    # Assuming the dataset is in a ZIP file
    zip_file = 'forest_fire_dataset.zip'
    unzip_data(zip_file, data_dir)

    # Split the dataset into training and testing
    split_dataset(data_dir)
    print("Dataset setup complete!")
    
Project Structure
The final project structure might look like this:

markdown
Copy code
forest_fire_detection/
│
├── dataset/
│   ├── fire/
│   └── no_fire/
│
├── naive_approach.py
├── setup_datasets.py
├── Forest_Fire_Prediction_using_Machine_Learning.txt
└── model/
    ├── cnn_model.py
    ├── train_model.py
    └── predict_fire.py

Modules Overview
setup_datasets.py: Handles dataset setup and preparation.
transfer_learning.py: Implements the transfer learning model based on InceptionV3, including the definition of a batch generator that applies data augmentation techniques. This file also manages training, including the option to freeze layers and adjust the learning rate for fine-tuning.
video_annotation.py: Annotates video frames with predictions from the trained CNN.
evaluate_model.py: Evaluates the model and extracts challenging examples for further training.
Usage
The project is controlled via a command-line interface (CLI) through the launcher.py module, which accepts various commands to manage training, fine-tuning, prediction, and video annotation. Here are some examples of the available commands:

Train the model:

css
Copy code
launcher.py train -data DATASET -prop PROPORTION -freeze FREEZE [-epochs EPOCHS] [-batch BATCH_SIZE]
Fine-tune a pre-trained model:

css
Copy code
launcher.py tune -model MODEL_PATH -lr LEARNING_RATE -data DATASET -prop PROPORTION -freeze FREEZE [-epochs EPOCHS] [-batch BATCH_SIZE]
Make predictions on an image:

lua
Copy code
launcher.py predict -path IMAGE_PATH -model MODEL_PATH
Extract difficult examples:

kotlin
Copy code
launcher.py extract -data DATASET -model MODEL_PATH -threshold EXTRACT_THRESHOLD
Evaluate the model on a test set:

bash
Copy code
launcher.py test -data DATASET -model MODEL_PATH
Annotate a video with predictions:

css
Copy code
launcher.py video -in INPUT_VIDEO_PATH -out OUTPUT_VIDEO_PATH -model MODEL_PATH [-freq FREQ]
Model and Performance
The trained model is saved as transfer_learned_model.h5 at the root of the project. The model's performance, measured by categorical loss and accuracy, is as follows:

On the provided test set (100 samples):

Loss: 0.3497
Accuracy: 91.09%
On the entire dataset (5953 samples):

Loss: 0.0361
Accuracy: 99.14%
From our experiments, we found that the model performs exceptionally well at classifying 'fire' and 'no fire' images with high accuracy. However, the 'start fire' images are more challenging to classify. This could be due to the fact that 'fire' images may also contain smoke, and 'start fire' images sometimes feature small flames that are harder to distinguish.

Video Examples
Examples of annotated videos processed by the model can be found in the video_examples/ directory. These videos demonstrate the model's ability to make real-time predictions on video frames.








Video examples
Examples of videos annotated by our model can be found below.

https://github.com/akshaybannatti/Detecting-Forest-Fire-Convolutional-Neural-Network/assets/50884750/b305e6a7-3ad9-49b8-a7b8-8b853248eae7



https://github.com/akshaybannatti/Detecting-Forest-Fire-Convolutional-Neural-Network/assets/50884750/67c5614c-3883-42f7-acb2-6bc128dd24d2



https://github.com/akshaybannatti/Detecting-Forest-Fire-Convolutional-Neural-Network/assets/50884750/dfc821e7-7139-48b2-ad72-cd3b8d143f4d

![video_0 (1)](https://github.com/akshaybannatti/Detecting-Forest-Fire-Convolutional-Neural-Network/assets/50884750/378adb83-cc18-49d2-b365-2f14f2626ec2)  











