# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:51:59 2021

@author: zahil
"""

### Imports ###
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(1, 'E:\\Projects\\Utilities') # insert at 1, 0 is the script path (or '' in REPL)
from DemodDataset import DemodDataset

### Paths and consts ### 
ANNOTATIONS_FILE = 'E:\Projects\Flight\DLCode\Labels.csv'
AUDIO_FILE = 'E:\Projects\Flight\DLCode\HaifaDemoded.mat'
desired_label = 'HebOrEng'
SAMPLE_RATE = 12500
NUM_SAMPLES = 40000
device = "cuda"

### HyperParams ###
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.001
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64)
EmbeddingDim = 128

### DataSets ###
demod_ds = DemodDataset(ANNOTATIONS_FILE,AUDIO_FILE,desired_label,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,device)
train_size = int(0.9 * len(demod_ds))
test_size = len(demod_ds) - train_size
train_set, val_set = torch.utils.data.random_split(demod_ds, [train_size, test_size])
print(f"There are {len(train_set)} samples in the train set and {len(val_set)} in validation set.")
#signal, label = demod_ds[0]

### DataLoader ### 
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE) 
test_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)

### TBD: A neural network base encoder ###

### TBD: contrastive loss function
# https://www.google.com/search?q=contrastive+loss+function+pytorch&oq=contrastive+loss+function&aqs=chrome.1.69i59j0i22i30l5.2159j0j7&sourceid=chrome&ie=UTF-8
#loss_fn = nn.CrossEntropyLoss()
#optimiser = torch.optim.Adam(cnn.parameters(),lr=LEARNING_RATE)
    
#### TBD: A stochastic data augmentation module ###
#signal_a = 
#signal_b = 



