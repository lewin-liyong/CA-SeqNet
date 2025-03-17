import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

folder_path = '/RAID5/projects/fuxingwen/ly/class/non-artifact/'
train_folder = os.path.join(folder_path, 'train')
test_folder = os.path.join(folder_path, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all. npy files
files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
# The training set test set is randomly divided according to the proportion.
train_files, test_files = train_test_split(files, test_size=1/16, random_state=42)

# Move files to the corresponding folder
for file in train_files:
    shutil.move(os.path.join(folder_path, file), os.path.join(train_folder, file))
for file in test_files:
    shutil.move(os.path.join(folder_path, file), os.path.join(test_folder, file))
