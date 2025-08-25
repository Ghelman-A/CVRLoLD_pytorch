from statistics import mode
import comet_ml  # Had to add it due to an error with comet logger
from pytorch_lightning.loggers import CometLogger
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint, EarlyStopping

# Custom
from config import cvrl_config

from lightning_module import LitSupervisedAct
from equipmentDataModule import EquipmentDataModule


def main(config):
    pl.seed_everything(seed=1234, workers=True)
    tr_mode = config['train_mode']

    #------ Setting the lightning trainer
    dm = EquipmentDataModule(config)
    model = LitSupervisedAct(config)

    #------ Setting the Comet logger
    comet_logger = CometLogger(
        api_key="",
        workspace="",  # Optional
        project_name="",  # Optional
        experiment_name=f'{config["exp_datetime"]}'
    )
    comet_logger.log_graph(model)  # Record model graph
    
    #------ Setting the required callbacks
    lr_mon = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(f"{config['checkpoint_dir']}{config['exp_datetime']}", 
                                          monitor=config[tr_mode]['monitor'], mode=config[tr_mode]['mode'])
    early_stop = EarlyStopping(monitor=config[tr_mode]['monitor'], mode=config[tr_mode]['mode'], 
                               patience=config[tr_mode]['patience'],
                               min_delta=config[tr_mode]['min_delta'])

    #------ Training!
    sync_norm = True if tr_mode == 'SSL' else False
    replace_sampler = False if tr_mode == 'SSL' else True
    trainer = pl.Trainer(gpus=config['train_cfg']['gpu_device_ids'], strategy='ddp', logger=[comet_logger], log_every_n_steps=1,
                         min_epochs=config['train_cfg']['epochs'], max_epochs=600, sync_batchnorm=sync_norm, replace_sampler_ddp=replace_sampler,
                         callbacks=[lr_mon, ModelSummary(max_depth=100), checkpoint_callback, early_stop])
    
    trainer.fit(model, datamodule=dm)

    if tr_mode == "SSL":
        config["train_mode"] = tr_mode = "linear_eval"
        config["linear_eval"]["checkpoint_path"] = checkpoint_callback.best_model_path
        
        dm = EquipmentDataModule(config)
        eval_model = LitSupervisedAct(config)

        lr_mon = LearningRateMonitor(logging_interval='step')
        early_stop = EarlyStopping(monitor=config[tr_mode]['monitor'], mode=config[tr_mode]['mode'],
                                   patience=config[tr_mode]['patience'],
                                   min_delta=config[tr_mode]['min_delta'])
        checkpoint_callback = ModelCheckpoint(f"{config['checkpoint_dir']}{config['exp_datetime']}", monitor='train_loss', mode='min',
                                              filename='line_eval_epoch={epoch}-step={step}')

        val_trainer = pl.Trainer(gpus=config['train_cfg']['gpu_device_ids'], strategy='ddp', logger=[comet_logger],
                                 min_epochs=config['train_cfg']['epochs'], log_every_n_steps=1,                                 
                                 callbacks=[lr_mon, ModelSummary(max_depth=100), checkpoint_callback, early_stop],)
        val_trainer.fit(eval_model, datamodule=dm)
        
         #------ Testing only on one GPU
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            val_trainer = pl.Trainer(gpus=[1], logger=[comet_logger], log_every_n_steps=1)
            model = LitSupervisedAct.load_from_checkpoint(checkpoint_callback.best_model_path)

            val_trainer.test(model, datamodule=dm)
    
    else:
        torch.distributed.destroy_process_group()

        if trainer.is_global_zero:
            trainer = pl.Trainer(gpus=[1], logger=[comet_logger], log_every_n_steps=1)
            model = LitSupervisedAct.load_from_checkpoint(checkpoint_callback.best_model_path)

            trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main(cvrl_config)
