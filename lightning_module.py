import math
from typing import OrderedDict
from numpy import mat
import torch
from torch import Tensor, nn
import torch.nn.functional as func
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchmetrics import Accuracy

from models.contrastivemodel import CVRLModel
from yattag import Doc      # For sending data to html
from typing import Optional


class LitSupervisedAct(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config          
        self.CVRL_model = CVRLModel(config)
        self.cos = torch.nn.CosineSimilarity(dim=2)  # for computing InfoNCE loss
        if config['train_mode'] in ["semi", "linear_eval"]:
            self.load_pretrained_model(config[config["train_mode"]]['checkpoint_path'])

        self.save_hyperparameters()
        self.test_acc = Accuracy()              # (task='multiclass', num_classes=3) # For latest versions
        self.val_acc = Accuracy()               # (task='multiclass', num_classes=3) # For latest versions
        self.train_acc = Accuracy()             # (task='multiclass', num_classes=3) # For latest versions

    def training_step(self, batch, batch_idx):
        if self.config['train_mode'] == 'SSL':

            loss = self.shared_step(batch)
            self.log("ssl_train_loss", loss, on_step=True, on_epoch=True, batch_size=batch[0].shape[0])
        else:
            data, label = batch
            logits = self.CVRL_model(data)

            loss = func.cross_entropy(logits, label)
            self.train_acc(logits, label)

            self.log_dict({"train_loss": loss, "train accuracy": self.train_acc}, on_step=True, on_epoch=True,
                          batch_size=data.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        if self.config['train_mode'] != 'SSL':
            data_, label_, clip_frames = batch
            loss = 0
            preds = []
            logits_list = []
            for data, label in zip(data_, label_):
                logits = self.CVRL_model(data)
                logits = func.log_softmax(logits, dim=1).mean(dim=0, keepdim=True)
                logits_list.append(logits)

                preds.append(logits.argmax(dim=1).item())
                loss += func.nll_loss(logits, label).item()

            self.val_acc(torch.stack(logits_list).squeeze(dim=1), torch.stack(label_).squeeze())
            loss /= len(data_)

            # Logging everything!
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(data_))
            self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, batch_size=len(data_))

            return [label_, preds, clip_frames]

    def test_step(self, batch, batch_idx):
        data_, label_, clip_frames = batch
        loss = 0
        pred = []
        logits_list = []
        for data, label in zip(data_, label_):
            logits = self.CVRL_model(data)
            logits = func.log_softmax(logits, dim=1).mean(dim=0, keepdim=True)
            logits_list.append(logits)

            pred.append(logits.argmax(dim=1).item())
            loss += func.nll_loss(logits, label).item()

        self.test_acc(torch.stack(logits_list).squeeze(dim=1), torch.stack(label_).squeeze())
        loss /= len(data_)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(label_))
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, batch_size=len(label_))

        return [label_, pred, clip_frames]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config['train_cfg']['max_lr'], momentum=0.9)
        warmup_steps = self.num_training_steps * self.config['train_cfg']['warm_up_epochs']
        total_steps = self.num_training_steps * self.config['train_cfg']['epochs']
        lr_sch = CosineAnnealingWarmupRestarts(optimizer,
                                               first_cycle_steps=total_steps,
                                               max_lr=self.config['train_cfg']['max_lr'],
                                               min_lr=self.config['train_cfg']['min_lr'],
                                               warmup_steps=warmup_steps, gamma=self.config['train_cfg']['gamma'])
        scheduler = {
            "scheduler": lr_sch,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs) -> None:
        if self.config['train_mode'] == 'supervised':
            labels = []
            preds = []
            clip_frames = []
            for out in outputs:
                labels.extend([label.item() for label in out[0]])
                preds.extend(out[1])
                clip_frames.extend(out[2])

            self.logger.experiment.log_confusion_matrix(y_true=labels, y_predicted=preds,
                                                           title=f'Confusion matrix, Epoch {self.current_epoch}',
                                                           labels=self.config['OBJECT_ACTION_NAMES']['Excavator'])

            self.logger.experiment.log_html(self.result_html(clip_frames, labels, preds,
                                                                f'Validation Results, Epoch {self.current_epoch}'))

    def test_epoch_end(self, outputs) -> None:
        labels = []
        pred = []
        clip_frames = []

        for out in outputs:
            labels.extend([label.item() for label in out[0]])
            pred.extend(out[1])
            clip_frames.extend(out[2])

        self.logger.experiment.log_confusion_matrix(y_true=labels, y_predicted=pred,
                                                    title=f'Test confusion matrix',
                                                    labels=self.config['OBJECT_ACTION_NAMES']['Excavator'])

        self.logger.experiment.log_html(self.result_html(clip_frames, labels, pred, 'Test Results'))

    @staticmethod
    def result_html(clip_frames, labels, preds, header):
        """
        A temporary script for creating a table for the validation and test results
        to make comparison easier.
        :param clip_frames: The name of the first frame of the input video
        :param labels: The true labels
        :param preds: The prediction results
        :param header: The title of the created table
        :return:
        """
        doc, tag, text = Doc().tagtext()

        with tag('html'):
            with tag('head'):
                with tag('style'):
                    text('table, th, td {border: 1px solid black; border-collapse: collapse;}')

            with tag('body'):
                with tag('h2'):
                    text(header)
                with tag('table'):
                    with tag('tr'):
                        with tag('th'):
                            text('#')
                        with tag('th'):
                            text('Clip Frames')
                        with tag('th'):
                            text('Label')
                        with tag('th'):
                            text('Prediction')

                    for idx, res in enumerate(zip(clip_frames, labels, preds)):
                        with tag('tr'):
                            with tag('td'):
                                text(str(idx))
                            with tag('td'):
                                text(str(res[0]))
                            with tag('td'):
                                text(str(res[1]))
                            with tag('td'):
                                text(str(res[2]))

        return doc.getvalue()

    def load_pretrained_model(self, checkpoint_dir) -> None:
        """
        Args:
            checkpoint_dir (_type_): The directory of the saved SSL model.

        """
        
        if checkpoint_dir == self.config['checkpoint_dir']:
            print("\nNo pre-trained models specified! This will become supervised learning using only \
                  the specified portion of the dataset.\n")
            return
        
        #-----------------------------------------------------------------------------------#
        # Loading the pretrained model and modifying the checkpoint keys except the last fc #
        # layer, which will be replaced.                                                    #
        #-----------------------------------------------------------------------------------#
        saved_model = torch.load(checkpoint_dir)['state_dict']
        saved_model_new_keys = OrderedDict()
        for key, item in saved_model.items():
            if 'fc' not in key:
                saved_model_new_keys[key.replace('CVRL_model.', '')] = item
        
        #-----------------------------------------------------------------------------------#
        # Replacing the backbone with the loaded weights and making sure that the gradient  #
        # is only set for the case of fine-tuning in the semi-supervised part.              #
        #-----------------------------------------------------------------------------------#
        self.CVRL_model.load_state_dict(saved_model_new_keys, strict=False)
        for name, param in self.CVRL_model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = True if self.config['train_mode'] == 'semi' else False
        
    #################################################################
    #   Helper functions for the SSL case taken from pl bolt git!   #
    #################################################################
    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]

        Note: I have tested this loss a few times! It is correct!
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():

            ## The output of sll_gather: [world_size, batch_size, ...]
            out_1_dist = self.all_gather(out_1).view(-1, out_1.shape[-1])
            out_2_dist = self.all_gather(out_2).view(-1, out_2.shape[-1])

        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        batch_size = out_1.shape[0]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = self.cos(out.unsqueeze(1), out_dist.unsqueeze(0).contiguous())
        sim = torch.exp(cov / temperature)

        sim_i_j = torch.diag(sim[:, :2 * batch_size], batch_size)
        sim_j_i = torch.diag(sim[:, :2 * batch_size], -batch_size)
        pos = torch.cat([sim_i_j, sim_j_i], dim=0)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        neg = sim.sum(dim=-1)
        neg -= math.e ** (1 / temperature)
        neg = torch.clamp(neg, min=eps)  # clamp for numerical stability
        loss = -torch.log(pos / neg).mean()

        return loss

    def shared_step(self, batch):
        # final image in tuple is for online eval
        img1, img2 = batch

        # get h representations, bolts resnet returns a list
        h1 = self.CVRL_model(img1)
        h2 = self.CVRL_model(img2)

        return self.nt_xent_loss(h1, h2, self.config['train_cfg']['temperature'])

    @property
    def num_training_steps(self) -> int:
        """
            Total training steps inferred from datamodule and devices.
            Source: https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-774265729
        """
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        dataset = self.trainer._data_connector._train_dataloader_source.dataloader()
        batches = len(dataset)
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices

        # This line gives the total number, what I want is the per epoch one
        # return (batches // effective_accum) * self.trainer.max_epochs
        if self.config['train_mode'] == 'SSL':
            return batches
        else:
            return batches // effective_accum
