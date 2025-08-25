"""
    Author: Ali Ghelmani,           Date: June 19, 2022
"""

import matplotlib.pyplot as plt
import math
import torch

from config import cvrl_config
from equipmentDataModule import EquipmentDataModule


class VisualizeData:
    def __init__(self, config):
        self.config = config
        self.dm = EquipmentDataModule(config, vis_mode=True)

    def vis_train_data(self):
        self.dm.setup("fit")
        batch = next(iter(self.dm.train_dataloader()))

        self.visualize_batch(batch, max_image=9)

    def viz_val_data(self):
        self.dm.setup("fit")
        batch = next(iter(self.dm.val_dataloader()))

        data = torch.stack([clip[0] for clip in batch[0]])
        label = []
        for clip_info in zip(batch[1], batch[2]):
            label.append(f'{clip_info[0]}, {clip_info[1].name}')

        self.visualize_batch((data, label))

    @staticmethod
    def visualize_batch(data, max_image=16):
        """
        :param data: Batch data
        :param max_image: Maximum number of data in the batch to visualize
        :return:
        """
        vid, label = data

        max_image = min(max_image, vid.shape[0])
        row = int(math.sqrt(max_image))
        col = row if pow(row, 2) == max_image else row + 1

        _, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)

        row_idx = 0
        col_idx = 0

        for idx in range(max_image):
            axes[row_idx][col_idx].imshow(vid[idx][:, 0].permute(1, 2, 0))
            axes[row_idx][col_idx].set_title(f'label: {label[idx]}')

            col_idx += 1
            if col_idx == col:
                row_idx += 1
                col_idx = 0

        plt.show()


if __name__ == "__main__":
    viz_module = VisualizeData(cvrl_config)
    viz_module.vis_train_data()
    viz_module.viz_val_data()
