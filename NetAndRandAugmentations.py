# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:46:12 2021

@author: zahil
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from MyGeneralResNet18 import MyResNet18

##########################################################
# Random Augmentations:
# 1) AWGN to all spectogram 
# 2) Random shift in the time axis of about ~0.3 sec (done with RandomAffine)
# 3) Random shift in the frequency axis of about 1000Hz (done with RandomAffine)
#https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=3.): #This WGN is added to the spectogram (dB units) rather than the signal. It has nothing to do with the termal noise!!
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        device = tensor.device
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


##########################################################
# A net for the semi-supervised training:
    
class MyEmbeddingAndProjectionNet(nn.Module):     
        
    def __init__(self, embedding_size=32):
        super().__init__()
        
        self.embedding = MyResNet18(InputChannelNum=1,IsSqueezed=1,LastSeqParamList=[512,64],pretrained=True)
        
        self.projection = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.Tanh()
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, x):
        embedding = self.calculate_embedding(x)
        projection = self.projection(embedding)
        return embedding, projection

##########################################################
# A net for the supervised training 
# Note that the forward is actually using only the "calculate_embedding" part and not the "projection" part, as they did in SimClr paper
# I still not sure what is the point of using my "SecondNet" instead of the "projection"(which is discarded), but this is what they suggested in the paper..
# See more details in the SimCLR paper or at:
# https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
    
class InferenceNet(nn.Module):     
        
    def __init__(self,PretrainedFile=None):
        super().__init__()
        self.FirstNet = MyEmbeddingAndProjectionNet()
        if PretrainedFile is not None:
            self.FirstNet.load_state_dict(torch.load(PretrainedFile))
            print('loaded weights from ' + PretrainedFile)
        
        # I will not use the "projection" net as SimClr suggested
        self.SecondNet = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        embedding = self.FirstNet.calculate_embedding(x)
        projection = self.SecondNet(embedding)
        return projection