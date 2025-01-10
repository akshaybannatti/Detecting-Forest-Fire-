# Detecting-Forest-Fire-

Forest fire detection using CNN
This project is an attempt to use convolutional neural networks (CNN) to detect the presence or the start of a forest fire in an image. The idea is that this model could be applied to detect a fire or a start of a fire from (aerial) surveillance footage of a forest. The model could be applied in real-time to low-framerate surveillance video (with fires not moving very fast, this assumption is somewhat sound) and give alert in case of fire.

A description of the project, along with examples of video annotation by our network is provided below.

![video_0 (1)](https://github.com/akshaybannatti/Detecting-Forest-Fire-Convolutional-Neural-Network/assets/50884750/378adb83-cc18-49d2-b365-2f14f2626ec2)  

Dataset Overview
The model was trained using a dataset containing images from three categories: 'fire', 'no fire', and 'start fire'. In total, the dataset comprises around 6,000 images, primarily featuring forest or forest-like environments.

'Fire' images contain visible flames.
'Start fire' images feature smoke, indicating the beginning of a fire.
'No fire' images depict forest scenes without any fire or smoke.
Data Augmentation
Through experimentation, we observed that the model struggled to classify 'start fire' images effectively. To address this, we enriched the dataset by extracting frames from videos that depict the early stages of a fire. To improve the model’s ability to generalize, we applied data augmentation techniques provided by Keras. This involved performing random transformations on the images, including zooms, shifts, crops, and rotations, to diversify the training data before it was fed into the model.

Project Structure
The project was designed to be clean and organized, covering all aspects of CNN creation and training. The directory structure is as follows:

arduino
Copy code
├── launcher.py
├── transfer_learning.py
├── video_annotation.py
├── evaluate_model.py
├── transfer_learned_model.h5
├── setup/
│   ├── setup_datasets.py
│   └── naive_approach.py
├── notebooks/
│   ├── datasets_analysis.ipynb
│   └── performance_analysis.ipynb
├── custom_model/
│   ├── cladoh.py
│   ├── model_train_test.py
│   └── model_train_tester.py
├── video_examples/
│   ├── video_0.gif
│   ├── video_1.gif
│   ├── video_2.gif
│   └── video_3.gif
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











