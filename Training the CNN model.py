import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import create_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint

def setup_image_generators(train_dir, test_dir, batch_size=32, target_size=(224, 224)):
    # Image data augmentation for training
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # For validation data (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training and testing data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, test_generator

def train_model(train_dir, test_dir, model_save_path='forest_fire_model.h5'):
    # Set up image generators
    train_generator, test_generator = setup_image_generators(train_dir, test_dir)

    # Create the CNN model
    model = create_cnn_model()

    # Define a checkpoint to save the best model
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        callbacks=[checkpoint])

    # Save the final model
    model.save(model_save_path)

    print(f"Model training complete. Saved to {model_save_path}")

if __name__ == "__main__":
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'

    # Train the model
    train_model(train_dir, test_dir)
