# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 22:00:31 2021

@author: zahil
"""
# This is a one-time run to split the dataset and save the splitting for better reproducibility

import numpy as np
import torch
from DemodDataset import DemodDataset , ProcessedDataset
demod_ds = ProcessedDataset('ProcessedTorchData.pt',label_ind = 1) #calling the after-processed dataset
un_labeled_inds = np.where(torch.isnan(demod_ds.labels))[0]
labeled_inds = np.where(~torch.isnan(demod_ds.labels))[0] #around 167 samples
labeled_half_len = np.int64(len(labeled_inds)/2)
np.random.shuffle(labeled_inds)
val_inds = labeled_inds[labeled_half_len:]
train_inds  =  np.concatenate((labeled_inds[:labeled_half_len],un_labeled_inds))
torch.save([train_inds,val_inds],'RandIndsSplit.pt')