# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:29:27 2021

@author: zahil
"""

import numpy as np
import torch
import torchaudio
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(1, 'E:\\Projects\\Utilities') # insert at 1, 0 is the script path (or '' in REPL)
from MyMetrics import ContrastiveLoss 
from DemodDataset import DemodDataset , ProcessedDataset
from NetAndRandAugmentations import AddGaussianNoise,MyEmbeddingAndProjectionNet
import time