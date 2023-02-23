# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:52:19 2021

@author: zahil
"""
# Here I created 2 dataset:
# 1. The DemodDataset - which works directly with Mat file I saved in matlab (The mat file was created by reading the dat files, pad\cut them)
#    This DS enable us to choose the params for the MEL spectogram on the fly,but the The problem is that it takes a lot of time to process a single sample.
#    Therefore I created another DS
# 2. The ProcessedDataset - Read the file "ProcessedTorchData" which was created by a "one time run" script. basically it saved the sample after pre-process

#import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import scipy.io as sio

class ProcessedDataset(Dataset):
    def __init__(self,Path,label_ind = 1):
        [self.tensor, label_1, label_2, label_3, label_4] = torch.load(Path)
        if label_ind == 1:
            self.labels = label_1
        if label_ind == 2:
            self.labels = label_2
        if label_ind == 3:
            self.labels = label_3
        if label_ind == 4:
            self.labels = label_4
        self.Len = self.tensor.shape[0]
    def __len__(self):
        return self.Len
    def __getitem__(self, idx):
        signal = self.tensor[idx,::] 
        label = self.labels[idx] 
        return signal,label

class DemodDataset(Dataset):

    def __init__(self,
                 annotations_file = 'E:\Projects\Flight\DLCode\Labels.csv',
                 audio_file = 'E:\Projects\Flight\DLCode\HaifaDemoded.mat',
                 desired_label = 'HebOrEng',
                 transformation = torchaudio.transforms.MelSpectrogram(sample_rate=12500,n_fft=1024,hop_length=512,n_mels=64),
                 sample_rate = 12500,
                 num_samples = 40000,
                 device = "cpu"):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_mat = sio.loadmat(audio_file)["DemodedMat"] #for now it is 3.2 second of burst sampled at 12.5Khz = 40,000 samples
        self.desired_label = desired_label #Actually i am not using this
        self.device = device
        self.transformation = transformation.to(self.device)
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_1 = self.annotations.iloc[index, 1] #HebOrEng
        label_2 = self.annotations.iloc[index, 2] #GndOrAir
        label_3 = self.annotations.iloc[index, 3] #CivOrArmy
        label_4 = self.annotations.iloc[index, 4] #HaifaOrPluto
        signal = torch.tensor(self.audio_mat[index,:],dtype=torch.float32)
        signal = signal.to(self.device)
        #signal = self._resample_if_necessary(signal, sr) #done in matlab
        #signal = self._mix_down_if_necessary(signal)     #done in matlab 
        #signal = self._cut_if_necessary(signal)          #done in matlab
        #signal = self._right_pad_if_necessary(signal)    #done in matlab
        signal = self.transformation(signal)
        signal = 20.0*torch.log10(signal+1e-10)
        return signal, label_1, label_2, label_3, label_4

    # def _cut_if_necessary(self, signal):
    #     if signal.shape[1] > self.num_samples:
    #         signal = signal[:, :self.num_samples]
    #     return signal

    # def _right_pad_if_necessary(self, signal):
    #     length_signal = signal.shape[1]
    #     if length_signal < self.num_samples:
    #         num_missing_samples = self.num_samples - length_signal
    #         last_dim_padding = (0, num_missing_samples)
    #         signal = torch.nn.functional.pad(signal, last_dim_padding)
    #     return signal

    # def _resample_if_necessary(self, signal, sr):
    #     if sr != self.target_sample_rate:
    #         resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
    #         signal = resampler(signal)
    #     return signal

    # def _mix_down_if_necessary(self, signal):
    #     if signal.shape[0] > 1:
    #         signal = torch.mean(signal, dim=0, keepdim=True)
    #     return signal
    
    
#####  It is good valerio implemented a "main" function so we can test the dataset directly without help of external script 
   
if __name__ == "__main__":
    ANNOTATIONS_FILE = 'E:\Projects\Flight\DLCode\Labels.csv'
    AUDIO_FILE = 'E:\Projects\Flight\DLCode\CombinedDemoded.mat'
    desired_label = 'HebOrEng'
    SAMPLE_RATE = 12500
    NUM_SAMPLES = 40000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    demod_ds = DemodDataset(ANNOTATIONS_FILE,
                            AUDIO_FILE,
                            desired_label,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(demod_ds)} samples in the dataset.")
    signal, label_1, label_2, label_3, label_4 = demod_ds[0]