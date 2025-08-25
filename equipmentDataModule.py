"""
    This scripts wraps most of the data processing steps into a LightningDataModule for ease
    of reproducibility and use.

    Author: Ali ghelmani,       Last Modified: June 12, 2022
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvideotransforms import video_transforms, volume_transforms

from preprocessing.customdatasets import CvRLDataset, SupervisedDataset
from preprocessing.dataprocessing import PrepareDataLists
from custom_sampler import CustomBatchSampler, CustomTestSampler


class EquipmentDataModule(pl.LightningDataModule):
    def __init__(self, config, vis_mode=False):
        super().__init__()
        self.config = config
        self.prepare_data()

        self.vis_mode = vis_mode  # used to get the batch data using a single process to visualize
        self.num_workers = 0 if self.vis_mode else 4

        self.train_transform = self.train_transforms()
        self.test_transform = self.test_transforms()

        # Assigning the path for the train and evaluation files
        self.train_mode = config['train_mode']
        self.train_pkl = config[self.train_mode]['train_pkl_dir']
        self.val_pkl = config[self.train_mode]["eval_pkl_dir"]
        self.test_pkl = config["test_pkl_dir"]

        # Variables that will be initiated in the setup method
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        if self.config['data_prep']['prep_needed']:
            data_transform = PrepareDataLists(self.config)
            data_transform.ssl_train()
            data_transform.sup_train()
            data_transform.lin_eval()
            data_transform.semi_train_eval()
            data_transform.test_n_sup_eval()

    def setup(self, stage: str = None):

        if stage in (None, 'fit'):
            if self.train_mode == "SSL":
                self.train_ds = CvRLDataset(self.config, self.train_pkl, self.train_transform)
            else:
                self.train_ds = SupervisedDataset(self.config, self.train_pkl, self.train_transform, train=True)
                self.val_ds = SupervisedDataset(self.config, self.test_pkl, self.test_transform, train=False)
                
        elif stage in (None, 'test'):
            self.test_ds = SupervisedDataset(self.config, self.test_pkl, self.test_transform, train=False)

    def train_dataloader(self):
        if self.train_mode == 'SSL':
            sampler = DistributedSampler(self.train_ds, drop_last=False)
            batch_sampler = CustomBatchSampler(self.config['SSL']['comb_count'], sampler, 
                                               self.config['train_cfg']['train_batch_size'], drop_last=True)
            return DataLoader(self.train_ds, batch_sampler=batch_sampler, num_workers=self.num_workers, collate_fn=self.ssl_collate)
        else:
            return DataLoader(self.train_ds, self.config['train_cfg']['train_batch_size'], shuffle=True,
                              drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds != None:
            return DataLoader(self.val_ds, self.config['train_cfg']['val_batch_size'], drop_last=False, 
                              sampler=CustomTestSampler(self.val_ds),
                              num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.config['train_cfg']['val_batch_size'], shuffle=False, drop_last=False,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        pass

    def train_transforms(self):
        in_size = self.config['aug_cfg']['resize_size']
        crop_scale = self.config['aug_cfg']['crop_area_ratio']
        aspect_ratio = self.config['aug_cfg']['crop_aspect_ratio']

        transform = [video_transforms.RandomResizedCrop(size=(in_size, in_size), scale=crop_scale, ratio=aspect_ratio),
                     video_transforms.RandomHorizontalFlip(p=self.config['aug_cfg']['flip_chance'])]

        if self.config['train_mode'] == 'SSL':
            br = self.config['aug_cfg']['jitter_brightness']
            ctr = self.config['aug_cfg']['jitter_contrast']
            sat = self.config['aug_cfg']['jitter_saturation']
            hue = self.config['aug_cfg']['jitter_hue']

            transform += [video_transforms.ColorJitter(self.config['aug_cfg']['jitter_chance'], br, ctr, sat, hue),
                          video_transforms.RandomGrayscale(self.config['aug_cfg']['greyscale_chance']),
                          video_transforms.GaussianBlurring(self.config['aug_cfg']['gaussian_blur_kernel'])]

        transform += [volume_transforms.ClipToTensor(),
                      video_transforms.Normalize(self.config['aug_cfg']['mean'], self.config['aug_cfg']['std'])]

        return video_transforms.Compose(transform)

    def test_transforms(self):

        in_size = self.config['aug_cfg']['resize_size']
        transform = [video_transforms.Resize(size=(in_size, in_size)),
                     volume_transforms.ClipToTensor(),
                     video_transforms.Normalize(mean=self.config['aug_cfg']['mean'],
                                                std=self.config['aug_cfg']['std'])]

        return video_transforms.Compose(transform)

    @staticmethod
    def collate_fn(data_list):
        """
        This function is used to batch the data for the eval case which contains a list of clips
        per single label. The batching is done by concatenating the data along the first dimension
        instead of the usual stacking in PyTorch which adds a new dim.

        :param data_list: A list of (data, label) tuples
        :return:
        """
        clip_list = []
        label_list = []
        clip_frames = []

        for data in data_list:
            clip_list.append(torch.stack(data[0], dim=0))
            label_list.append(torch.tensor(data[1]).reshape(1))
            clip_frames.append(data[2])

        return clip_list, label_list, clip_frames

    @staticmethod
    def ssl_collate(data_list):
        """
            It is assumed that the data is a list of tuples for two augmentations
            of each clip
        :param data_list:
        :return:
        """
        data1 = torch.stack([data[0] for data in data_list], dim=0)
        data2 = torch.stack([data[1] for data in data_list], dim=0)
        return data1, data2
