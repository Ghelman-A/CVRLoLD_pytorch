"""
    This script converts the Excavator dataset to h5py format for faster access during
    model training and inference.

    There are two outputs in this script:
        - The entire dataset loaded in cv2 format and converted to size 448 x 448 in h5py format
        - A dict file that has the idx, label, and the frame number for the data in the h5py file.
    Author: Ali Ghelmani,       Date: Jan. 24, 2022
"""
import h5py
from pathlib import Path
import numpy as np
import cv2
from config import cvrl_config
import pickle

ds_dir = Path(cvrl_config.raw_vid_dir)

image_list = []
index_dict = dict()

for idx, folder in enumerate(sorted(ds_dir.iterdir())):
    print(idx, folder.name)

    for frame in sorted(list(folder.glob("*.PNG"))):
        index_dict[str(frame)] = len(image_list)
        img = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(448, 448))
        image_list.append(img)

ds_array = np.stack(image_list, axis=0)
print(f"ds_array shape: {ds_array.shape}")

# Writing the created list to a h5py file
f = h5py.File(f'{cvrl_config["ds_list_dir"]}Excavator_ds.hdf5', 'w', libver='latest')
dset = f.create_dataset('dset', data=ds_array)
f.close()

with open(f"{cvrl_config['ds_list_dir']}Excavator_dset_idx.pkl", 'wb') as file:
    pickle.dump(index_dict, file)
