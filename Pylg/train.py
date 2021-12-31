#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
import os
logger = logging.getLogger(__name__)

def to_categorical(y,classes):
    a = np.zeros(classes,dtype=np.int32)
    a[y] = 1
    return a

class TitanicDataSet(torch.utils.data.Dataset):
    def __init__(self,mode='train',data_dir='/Users/vinay/Projects/MLOPS/Pylg/data',split = [0.7,0.5]):
        self.mode = mode
        self.split = split
        if data_dir:
            self.data_dir = data_dir
        else:
            pwd = os.getcwd()
            self.data_dir = Path(pwd)/'data'
        self.data = self.get_split()
        self.input_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q',
       'Embarked_S']
        self.label_cols = ['label']

    def __getitem__(self,index):
        x = self.data[self.input_cols].iloc[index]
        y = self.data[self.label_cols].iloc[index]
        y_true = np.array(y,dtype=np.int32)
        return np.array(x,dtype=np.float32),y_true

    def __len__(self):
        return len(self.data)

    def get_split(self):
        data = pd.read_csv(self.data_dir/'data.csv')
        ids = np.random.permutation(len(data))
        train_id = int(len(ids)*self.split[0])
        valid_id = train_id + int((len(ids)-train_id-1)*self.split[-1])
        if self.mode == 'train':
            return data.iloc[ids[:train_id]]
        elif self.mode == 'valid':
            return data.iloc[ids[train_id:valid_id]]
        else:
            return data.iloc[valid_id:]


# Model class
def accuracy(y_hat,y):
    #import pdb;pdb.set_trace()
    return torch.mean((y_hat == y).type(torch.float32))


class Model(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.linear1 = nn.Linear(hparams.input_sz,hparams.ll1)
        self.linear2 = nn.Linear(hparams.ll1,hparams.ll2)
        self.outlayer = nn.Linear(hparams.ll2,2)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.outlayer(x)
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        #import pdb;pdb.set_trace()
        y_hat = self(x)
        y_true = y.squeeze().type(torch.LongTensor)
        loss = F.cross_entropy(y_hat,y_true)
        #print(f'batch indx:{batch_idx},{loss.item()}')
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        #import pdb;pdb.set_trace()
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y.squeeze())
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer


@hydra.main(config_path='config', config_name='default')
def train(cfg):
    # The decorator is enough to let Hydra load the configuration file.
     # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))

    # We recover the original path of the dataset:
    path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    #Load the data
    train_data = TitanicDataSet(data_dir=path,split=cfg.data.split)
    val_data = TitanicDataSet(mode='valid',data_dir=path,split=cfg.data.split)
    test_data = TitanicDataSet(mode='test',data_dir=path,split=cfg.data.split)

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size)

    pl.seed_everything(cfg.seed)
    #model
    net = Model(cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(net,train_loader,val_loader)

if __name__ == "__main__":
    train()




