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
