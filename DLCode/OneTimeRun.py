# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:08:05 2021

@author: zahil
"""

# one time run to save each of the samples in new dataset to avoid real time transformations

import torch
import torchaudio
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from DemodDataset import DemodDataset

### Paths and consts ### 
ANNOTATIONS_FILE = 'E:\Projects\Flight\DLCode\Labels.csv'
AUDIO_FILE = 'E:\Projects\Flight\DLCode\HaifaDemoded.mat'
desired_label = 'HebOrEng'
SAMPLE_RATE = 12500
NUM_SAMPLES = 40000
device = "cuda"

### HyperParams ###
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=400,n_mels=64)
EmbeddingDim = 128

### DataSets ###
demod_ds = DemodDataset(ANNOTATIONS_FILE,AUDIO_FILE,desired_label,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,device)

### DataLoader ### 
dataloader = DataLoader(demod_ds, batch_size=1)
 
X = torch.tensor([]).cuda()
Y = torch.tensor([])

for idx, [signal,label] in enumerate(dataloader):
    #torch.save([torch.squeeze(signal),label],'ProcessedTorchData' + '\\' + str(idx))
    #print(signal.shape) #torch.Size([1, 64, 101]) - i.e. torch is saving with the batch dim, altough it is a single dim
    X = torch.cat((X,signal),dim=0)
    Y = torch.cat((Y,label),dim=0)
    
torch.save([X,Y],'ProcessedTorchData')

### I ran this code offline to make sure the paradigma is correct:    
# torch.save([signal,label],'tmp')
# [signal2,label2] = torch.load('tmp')
# torch.eq(signal, signal2)
# torch.equal(signal, signal2)