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
