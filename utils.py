import pandas as pd
import pickle as pkl
import numpy as np
import cv2
from PIL import Image


def load_pkl(data_fp, sample_size):
    data_list = []
    pklfl = open(data_fp, "rb")
    i = 0
    while i < sample_size:
        try:
            data_list.append(pkl.load(pklfl))
            print(len(data_list))
            i += 1
        except EOFError:
            break

    pklfl.close()
    data_list = pd.DataFrame(
        data_list, columns=["Genre", "Name", "Scene", "Fp", "Data"]
    )
    return data_list
