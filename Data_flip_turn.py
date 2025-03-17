import numpy as np
import os

folder_path = '/RAID5/projects/ly/class/non-artifact'

for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        neg_data = -data
        new_filename = f"{filename[:-4]}_overturn.npy"
        np.save(os.path.join(folder_path, new_filename), neg_data)