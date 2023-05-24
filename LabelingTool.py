# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:47:24 2023

@author: zahil
"""
### Imports
import numpy as np
import os.path as osp
import csv
import scipy.io as sio
import csv
import random
#import winsound
import sounddevice as sd


def PlaySound(sound_array):
    # Scale the data to the range [-1, 1]
    scaled_sound = 2 * ((sound_array - np.min(sound_array)) / (np.max(sound_array) - np.min(sound_array))) - 1 
    sd.play(scaled_sound,12.5e3)
    
    
### Init (feel free to change manually)
your_classes = {'0': 0, '1': 1, 'n': -1} # 'n' is for "not sure"
quit_key = 'q'
column_to_label = 'HebOrEng'  #'HebOrEng' , 'GndOrAir' , 'CivOrArmy' , 'HaifaOrPluto'
DataDir = 'Data'
csv_path = osp.join(DataDir,'Labels.csv')
audio_file = osp.join(DataDir,'CombinedDemoded.mat')

### loading the audio matrix (takes some time) 
audio_mat = sio.loadmat(audio_file)["DemodedMat"] #for now it is 3.2 second of burst sampled at 12.5Khz = 40,000 samples
print("loaded the audio mat")

### Loading the old excel (happening only one time) and reoepning it for appending
data = {}
if osp.exists(csv_path):
    with open(csv_path) as f:
        lines = f.read().splitlines()
    f.close()
    for idx,line in enumerate(lines):
        if idx == 0:
            header_line = line
            column_to_label_idx = header_line.split(',').index(column_to_label) 
            print('IMPORTANT: you have choosen to label the target:',line.split(',')[column_to_label_idx])
            continue
        #sample_index,HebOrEng,GndOrAir,CivOrArmy,HaifaOrPluto = line.split(',')
        sample_index = line.split(',')[0]
        values = line.split(',')[1:]
        data[sample_index] = values

### Creating a list of not-tagged elements
not_tagged_yet_list = []
for sample_index,val in data.items():
    if val[column_to_label_idx-1] == '':
        not_tagged_yet_list.append([sample_index,val])
print(f'There are {len(not_tagged_yet_list)} untagged samples\n')

### main tagging loop
while len(not_tagged_yet_list)>0:
    sample_index,val = random.choice(not_tagged_yet_list)
    not_tagged_yet_list.remove([sample_index,val])
    PlaySound(audio_mat[int(sample_index),:])
    print(f'\nSounding sample number {sample_index}. Please Insert key: 0 , 1 ,n to skip, or q to quit') 
    key = input("Please insert a key: ").strip()
    if key==quit_key:
        break
    elif key in your_classes.keys():
        data[sample_index][column_to_label_idx-1] = your_classes[key]
    else:
        print(f'invalid key code: {key}')
# +1 because the last sample was removed altough we did not tagged it (we probably pressesd q)
print(f'There are {len(not_tagged_yet_list)+1} untagged samples\n')

### Rewriting data to the excel
f = open(csv_path, 'w', newline='')
f.write(header_line + '\n')
for sample_index,val in data.items():
    s = f'{sample_index},{val[0]},{val[1]},{val[2]},{val[3]}' + '\n'
    f.write(s)
f.close()







