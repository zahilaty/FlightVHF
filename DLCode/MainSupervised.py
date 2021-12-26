# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:18:47 2021

@author: zahil
"""

### Imports ###
from All_imports import *
from NetAndRandAugmentations import InferenceNet

### HyperParams ###
BATCH_SIZE = 3
EPOCHS = 100
LEARNING_RATE = 0.0003

### DataSets ###
#demod_ds = DemodDataset(ANNOTATIONS_FILE,AUDIO_FILE,desired_label,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,device)
demod_ds = ProcessedDataset('ProcessedTorchData.pt',label_ind = 1) #calling the after-processed dataset
[l1,l2] = torch.load('RandIndsSplit.pt') # we need to save the indexes so we wont have data contimanation
assert len(list(set(l1).intersection(l2))) == 0
train_set = torch.utils.data.Subset(demod_ds, l1[:83]) #Hard coded!! TBD! the samples that are tagged
val_set = torch.utils.data.Subset(demod_ds, l2)
print(f"There are {len(train_set)} samples in the train set and {len(val_set)} in validation set.")
#How to get single sample for testing:   signal, label = demod_ds[0] ; signal.cpu().detach().numpy() ; %varexp --imshow sig

### DataLoader ### 
train_dataloader = DataLoader(train_set, batch_size=len(train_set),drop_last=True) #I need a fixed size bacth for the constractive loss
#test_dataloader = DataLoader(val_set, batch_size=val_set.dataset.__len__())

######### Part A - train with out unsupervised pretraining
net_A = InferenceNet()
net_A = net_A.cuda()
loss_fn = nn.BCELoss(reduction='none')
loss_fn_val = nn.BCELoss(reduction='mean')
optimizer_A = torch.optim.Adam(net_A.parameters(),lr=LEARNING_RATE)
Costs_A = np.array([])
Costs_val_A = np.array([])

for Epoch in range(EPOCHS):
    for batch_i, [batch,label] in enumerate(train_dataloader):
        optimizer_A.zero_grad()
        p = net_A(batch)
        target = torch.reshape(label,(-1,1)).float().cuda()
        loss_A = loss_fn(p,target)
        loss_A = loss_A*(target.detach() + (1-target.detach())*15/83)
        loss_A = loss_A.mean()
        Costs_A = np.append(Costs_A,loss_A.cpu().detach().numpy())
        loss_A.backward()
        optimizer_A.step()
        print('[%d, %5d] loss: %.3f' %(Epoch + 1, batch_i + 1, Costs_A[-1]))
    #check ValAccuracy every epoch
    net_A.eval()
    with torch.no_grad():
        val_samples, val_labels = val_set[:]
        p_val = net_A(val_samples)
        target_val = torch.reshape(val_labels,(-1,1)).float().cuda()
        #accuracy = torch.sum((p_val>0.5) == torch.reshape(val_labels,(-1,1)).cuda())/len(val_set)
        loss_val = loss_fn_val(p_val,target_val)
        Costs_val_A = np.append(Costs_val_A,loss_val.cpu().detach().numpy())
    net_A.train()

######### Part B - train with out unsupervised pretraining
net_B = InferenceNet('MySimClR_Cost_2.9545719623565674.pth')
net_B = net_B.cuda()
loss_fn = nn.BCELoss(reduction='none') #Hard coded for now - the ratio of "1" to "0"
optimizer_B = torch.optim.Adam(net_B.parameters(),lr=LEARNING_RATE)
Costs_B = np.array([])
Costs_val_B = np.array([])

for Epoch in range(EPOCHS):
    for batch_i, [batch,label] in enumerate(train_dataloader):
        optimizer_B.zero_grad()
        p = net_B(batch)
        target = torch.reshape(label,(-1,1)).float().cuda()
        loss_B = loss_fn(p,target)
        loss_B = loss_B*(target.detach() + (1-target.detach())*15/83)
        loss_B = loss_B.mean()
        Costs_B = np.append(Costs_B,loss_B.cpu().detach().numpy())
        loss_B.backward()
        optimizer_B.step()
        print('[%d, %5d] loss: %.3f' %(Epoch + 1, batch_i + 1, Costs_B[-1]))
    #check ValAccuracy every epoch
    net_B.eval()
    with torch.no_grad():
        val_samples, val_labels = val_set[:]
        p_val = net_B(val_samples)
        target_val = torch.reshape(val_labels,(-1,1)).float().cuda()
        #accuracy = torch.sum((p_val>0.5) == torch.reshape(val_labels,(-1,1)).cuda())/len(val_set)
        loss_val = loss_fn_val(p_val,target_val)
        Costs_val_B = np.append(Costs_val_B,loss_val.cpu().detach().numpy())
    net_B.train()