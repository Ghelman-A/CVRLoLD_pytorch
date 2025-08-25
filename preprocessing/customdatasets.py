from __future__ import print_function, division

import cv2
from torch.utils.data import Dataset
import random
import pickle
import h5py


class CvRLDataset(Dataset):
    def __init__(self, config, pkl_file, transform):
        self.config = config
        self.transform = transform

        if config['SSL']['all_ssl_comb']:
            assert '_full_comb' in pkl_file, "all_ssl_comb flag is set to True but the wrong pickle file is loaded!"

        with open(pkl_file, 'rb') as f:
            self.dataset_df = pickle.load(f)

        if config['train_cfg']['load_ds_to_mem']:
            f = h5py.File(self.config['ds_hdf5_dir'], 'r')
            self.loaded_ds = f['dset']

        self.ssl_headers = self.config['data_prep']['ssl_data_headers']

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if not self.config['SSL']['all_ssl_comb']:
            ##### SSL data is a list of clips from each video
            data_list = self.dataset_df['data'][idx]

            ##### For now uniform sampling
            clips = random.sample(range(len(data_list)), 2)

            data_0 = self.transform(self.load_clip(data_list[clips[0]]))
            data_1 = self.transform(self.load_clip(data_list[clips[1]]))
        else:
            pos_pair = self.dataset_df['data'][idx]
            
            data_0 = self.transform(self.load_clip(pos_pair[0]))
            data_1 = self.transform(self.load_clip(pos_pair[1]))

        return data_0, data_1
    
    def load_clip(self, clip):
        """
        This method reads the clip frames from the input path_list and returns the result in a list
        :param clip:
        :return: list of clip frames
        """
        if self.config['train_cfg']['load_ds_to_mem']:
            # In this case the clip has the indices in the h5py dset
            loaded_clip = [self.loaded_ds[idx] for idx in clip]
        else:
            raise TypeError("Only loading from h5py is implemented for now!")

        return loaded_clip


class SupervisedDataset(Dataset):
    """
        The purpose of this class is to provide labeled data for supervised,
        semi-supervised, and linear evaluation cases.
    """
    def __init__(self, config, pkl_file, transform, train=True):
        self.config = config
        self.supervised_headers = config['data_prep']['supervised_data_headers']
        self.transform = transform
        self.train_mode = train

        with open(pkl_file, 'rb') as f:
            self.dataset_df = pickle.load(f)

        if config['train_cfg']['load_ds_to_mem']:
            f = h5py.File(self.config['ds_hdf5_dir'], 'r')
            self.loaded_ds = f['dset']

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):

        if self.train_mode:

            data = self.load_clip(self.dataset_df['data'][idx])
            label = self.dataset_df['label'][idx]

            return self.transform(data), label
        else:
            """
                Eval data has a list of clips with one label for all of them!
            """
            data_list = self.dataset_df['data'][idx]
            data = [self.transform(self.load_clip(data_)) for data_ in data_list]
            label = self.dataset_df['label'][idx]
            clip_frames = self.dataset_df['frame_names'][idx][0][0]

            return data, label, clip_frames

    def load_clip(self, clip):
        """
        This method reads the clip frames from the input path_list and returns the result in a list
        :param clip:
        :return: list of clip frames
        """
        if self.config['train_cfg']['load_ds_to_mem']:
            # In this case the clip has the indices in the h5py dset
            loaded_clip = [self.loaded_ds[idx] for idx in clip]
        else:
            loaded_clip = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in clip]
        return loaded_clip
