# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 22:00:31 2021

@author: zahil
"""
#I run this ones and saved the indexes with torch.save

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

# train_size = int(0.9 * len(demod_ds))
# test_size = len(demod_ds) - train_size
# train_set, val_set = torch.utils.data.random_split(demod_ds, [train_size, test_size])
# torch.save([train_set.indices,val_set.indices],'RandIndsSplit.pt')

# Make sure this is OK with:
# list(set(train_set.indices).intersection(val_set.indices))