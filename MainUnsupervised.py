# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:51:59 2021
TODO:
    1) too much code duplication regarding "MainSupervised"
    2) need to verify the training loss again
@author: zahil
"""

### Imports ###
from All_imports import *

### HyperParams ###
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0003

### DataSets ###
#demod_ds = DemodDataset(ANNOTATIONS_FILE,AUDIO_FILE,desired_label,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,device)
demod_ds = ProcessedDataset('ProcessedTorchData.pt',label_ind = 1) #calling the after-processed dataset
[l1,l2] = torch.load('RandIndsSplit.pt') # already splitted to avoid data contamination
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
# TODO: fix this bug and rerun (transform get the mean and std you want to remove)
MyRandomTransforms = transforms.Compose([transforms.Normalize((0,), (1,)),transforms.RandomAffine(degrees = 0, translate=(0.2,0.15)),AddGaussianNoise(0., 1.)])

#### TODO: A stochastic data augmentation module ###
Costs = np.array([])

# For time profiler issues uncomment the following:
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
        # The same random transform is implemented to the entire batch
        sig_a = MyRandomTransforms(batch) 
        sig_b = MyRandomTransforms(batch) 
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
# TODO: plot Costs