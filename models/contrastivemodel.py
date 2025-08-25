import torch
from torchvision import models
import torch.nn as nn


class Identity(nn.Module):
    """ Identity module used to remove projection head after training"""

    def forward(self, x):
        return x


class CVRLModel(nn.Module):
    """
        This class loads the 3D ResNet model and modifies the FC layer according to the
        MLP projection head specifications in the paper. Furthermore, the forward functionality
        is also implemented in here.
    """
    def __init__(self, config):
        super(CVRLModel, self).__init__()
        self.train_mode = config['train_mode']
        self.model_cfg = config[self.train_mode]
        self.base_model = models.video.r3d_18(pretrained=config['pre_trained_resnet'])

        if self.train_mode in ['supervised', "linear_eval", "semi"]:
            self.base_model.fc = nn.Sequential(
                nn.Linear(512, self.model_cfg['in_layer_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.model_cfg['in_layer_size'], self.model_cfg['out_layer_size']),
            )
        else:
            self.base_model.fc = nn.Sequential(
            nn.Linear(512, self.model_cfg['in_layer_size']),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_cfg['in_layer_size'], self.model_cfg['out_layer_size']),
        )

        self.create_model()

    def create_model(self):
        return self.base_model

    def print_model(self):
        print(self.base_model)

    def forward(self, x):
        if self.train_mode in ('linear_eval', 'supervised'):
            x = self.base_model.stem(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = self.base_model.avgpool(x)
            x = x.squeeze()
            x_mean = torch.mean(x, dim=-1, keepdim=True)
            x_std = torch.std(x, dim=-1, keepdim=True)
            x = (x - x_mean) / x_std
            output = self.base_model.fc(x)
        else:
            output = self.base_model(x)
        return output
