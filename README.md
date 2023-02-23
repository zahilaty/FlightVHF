# SimCLR for Aircraft Radio Communications
This repository contains a partial implementation of SimCLR, a semi-supervised learning technique,
applied to a unique dataset of VHF communication between aircraft pilots that was collected during 2021.  
Some code is missing (or pointing on local paths), and it will uploaded at a later time.

![Mel spectogram of an audio record, transmitted in the VHF band at 2021](https://github.com/zahilaty/FlightVHF/blob/main/Images/MelSpecExample.jpg)

# Dataset
The dataset was collected with commercial COTS components: Raspberry Pi 3b, RTL-SDR dongle and VHF antenna (total cost < 100$).  
A GNU Radio block that contains channalizer and spectral power detection module recorded samples for offline processing. 
About 150 records out of ~8300 were labeled (see Labels.csv).  
The dataset is not publicly available, as it was specifically collected for this project.

# RF to audio
The RF_to_Audio.m code performs essential signal processing tasks such as DC removal,
frequency shifting, filtering, AM demodulation, decimation, and other standard techniques.
Next, the signal is trimmed or padded to a pre-defined length that was determined to accommodate
most typical record lengths.   
The complete code for these steps will be made available at a later time.

# Audio pre-process
I have created two datasets for this project:  
1) The DemodDataset, that load the Mat file that was created at RF_to_audio.m
and processes the audio samples using torchaudio. This dataset enables us to adjust
the parameters for the MEL spectrogram in real-time. However, processing a single sample
can be time-consuming.  
2) The ProcessedDataset, which reads the "ProcessedTorchData" file created by a one-time script.
This script runs the DemodDataset pre-processing and saves the sample.
The ProcessedDataset can be faster to use than the DemodDataset, 
but the pre-processing is not adjustable in real-time.

# Deep Learning Models
After studying the SimCLR paper, I developed a ResNet-18 model to serve as the embedding encoder,
which is then followed by a small fully connected projection network.  
The combined network was pretrained in an unsupervised manner (as explained in the paper).
However, the projection network was discarded and replaced with a new fully connected head.  
An experiment was conducted to evaluate the impact of unsupervised pretraining of the embedding encoder.

# Results
Unfortunately, it seems that the unsupervised pretraining made no difference,
at least in my data...

# Useful links:
https://www.youtube.com/c/ValerioVelardoTheSoundofAI?app=desktop  
https://www.rtl-sdr.com/sdrsharp-users-guide/  
https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/  
https://kevinmusgrave.github.io/pytorch-metric-learning/distances/

# License
This project is licensed under the MIT License. See the LICENSE file for more information.
