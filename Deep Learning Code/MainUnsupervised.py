# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:51:59 2021

@author: zahil
"""

### Imports ###
from All_imports import *

### Paths and consts ### 
#ANNOTATIONS_FILE = 'E:\Projects\Flight\DLCode\Labels.csv'
#AUDIO_FILE = 'E:\Projects\Flight\DLCode\HaifaDemoded.mat'
#desired_label = 'HebOrEng'
#SAMPLE_RATE = 12500
#NUM_SAMPLES = 40000
#device = "cuda"

### HyperParams ###
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0003
#mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=400,n_mels=64)

### DataSets ###
#demod_ds = DemodDataset(ANNOTATIONS_FILE,AUDIO_FILE,desired_label,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,device)
demod_ds = ProcessedDataset('ProcessedTorchData.pt',label_ind = 1) #calling the after-processed dataset
[l1,l2] = torch.load('RandIndsSplit.pt') # we need to save the indexes so we wont have data contimanation
assert len(list(set(l1).intersection(l2))) == 0
train_set = torch.utils.data.Subset(demod_ds, l1)
val_set = torch.utils.data.Subset(demod_ds, l2)
print(f"There are {len(train_set)} samples in the train set and {len(val_set)} in validation set.")
#How to get single sample for testing:   signal, label = demod_ds[0] ; signal.cpu().detach().numpy() ; %varexp --imshow sig

### DataLoader ### 
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE,drop_last=True) #I need a fixed size bacth for the constractive loss
test_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)

### A neural network base encoder ###
net = MyEmbeddingAndProjectionNet()
net = net.cuda()
net.load_state_dict(torch.load('MySimClR_Cost_2.957448959350586.pth'))

### contrastive loss function ###
# https://kevinmusgrave.github.io/pytorch-metric-learning/distances/
# https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
loss_fn = ContrastiveLoss(BATCH_SIZE)
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

### transform ###
# TBD - there is an error here -> transform get the mean and std you want to remove
MyRandomTransforms = transforms.Compose([transforms.Normalize((0,), (1,)),transforms.RandomAffine(degrees = 0, translate=(0.2,0.15)),AddGaussianNoise(0., 1.)])

#### TBD: A stochastic data augmentation module ###
Costs = np.array([])
# GetBatchTime = np.array([])
# TransformsTime = np.array([])
# ForwardTime = np.array([])
# LossTime = np.array([])
# BackwardTime = np.array([])

for Epoch in range(EPOCHS):
    t = time.time()
    for batch_i, [batch,label] in enumerate(train_dataloader):
        # GetBatchTime = np.append(GetBatchTime,time.time() - t)
        
        optimizer.zero_grad()
        #t = time.time()
        sig_a = MyRandomTransforms(batch) #The same random transform is implemented to the entire batch
        sig_b = MyRandomTransforms(batch) #The same random transform is implemented to the entire batch
        # TransformsTime = np.append(TransformsTime,time.time() - t)
        
        #t = time.time()
        projection_a = net(sig_a)[1] #index 1 is the projection
        projection_b = net(sig_b)[1] #index 1 is the projection
        # ForwardTime = np.append(ForwardTime,time.time() - t)
        
        #t = time.time()
        loss = loss_fn(projection_a,projection_b)
        # LossTime = np.append(LossTime,time.time() - t)
        Costs = np.append(Costs,loss.cpu().detach().numpy())
        
        #t = time.time()
        loss.backward()
        optimizer.step()
        # BackwardTime = np.append(BackwardTime,time.time() - t)
        
        t = time.time()
        print('[%d, %5d] loss: %.3f' %(Epoch + 1, batch_i + 1, Costs[-1]))

torch.save(net.state_dict(),'MySimClR_Cost_' + str(Costs[-1]) + '.pth')


### some figures
# plot Costs
# loss_fn(torch.randn((64,16)),torch.randn((64,16)))
# tmp1 = batch.detach().cpu().numpy()
# tmp2 = sig_a.detach().cpu().numpy()
# tmp3 = sig_b.detach().cpu().numpy()

# import matplotlib.pyplot as plt 
# plt.figure()
# plt.imshow(tmp1[16,:,:])
# plt.figure()
# plt.imshow(tmp2[16,:,:])
# plt.figure()
# plt.imshow(tmp3[16,:,:])
# plt.figure()
# plt.imshow(tmp4[16,:,:])