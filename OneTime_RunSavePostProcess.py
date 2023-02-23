# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:08:05 2021

@author: zahil
"""

# One time run to save each of the audio records in new dataset to avoid real time pre-processing

import torch
import torchaudio
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from DemodDataset import DemodDataset

### Paths and consts ### 
ANNOTATIONS_FILE = 'Data\Labels.csv'
AUDIO_FILE = 'Data\CombinedDemoded.mat'
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
Y1 = torch.tensor([])
Y2 = torch.tensor([])
Y3 = torch.tensor([])
Y4 = torch.tensor([])

for idx, [signal,label_1, label_2, label_3, label_4] in enumerate(dataloader):
    #torch.save([torch.squeeze(signal),label],'ProcessedTorchData' + '\\' + str(idx))
    #print(signal.shape) #torch.Size([1, 64, 101]) - i.e. torch is saving with the batch dim, altough it is a single dim
    X = torch.cat((X,signal),dim=0)
    Y1 = torch.cat((Y1,label_1),dim=0)
    Y2 = torch.cat((Y2,label_2),dim=0)
    Y3 = torch.cat((Y3,label_3),dim=0)
    Y4 = torch.cat((Y4,label_4),dim=0)
    
torch.save([X,Y1,Y2,Y3,Y4],'ProcessedTorchData.pt')

### You can run this code to make sure the saving does not harm the data:    
# torch.save([signal,label],'tmp')
# [signal2,label2] = torch.load('tmp')
# torch.eq(signal, signal2)
# torch.equal(signal, signal2)