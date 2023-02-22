# SimCLR for Aircraft Radio Communications
This repository contains a partial implementation of SimCLR, a semi-supervised learning technique,
applied to a unique dataset of VHF communication between aircraft pilots that was collected during 2021.  
Some code is missing, and it will uploaded at a later time.

![Mel spectogram of an audio record, transmitted in the VHF band at 2021](https://github.com/zahilaty/FlightVHF/blob/main/Example.jpg)

# Dataset
The dataset was collected with commercial COTS components: Raspberry Pi 3B, RTL-SDR dongle and VHF antenna (total cost < 100$).  
A GNU Radio block that contains channalizer and spectral power detection module recorded samples for offline processing. 
About 150 records out of ~8300 were labeled (see Labels.csv).  
The dataset is not publicly available, as it was specifically collected for this project.

# RF to audio
The necessary signal filtering, demodulation, and other steps were performed using standard AM demodulation techniques,
and the code for these steps will be uploaded at a later time

# Audio pre-process
Using torchaudio, mel spectograms were created "on the fly" (see DemodDataset.py)

# Results
The performance of the SimCLR model was compared to a straightforward supervised classifier.  
Unfortunately, it does not appear that SimCLR achieved better performance.

# Useful links:
https://www.youtube.com/c/ValerioVelardoTheSoundofAI?app=desktop  
https://www.rtl-sdr.com/sdrsharp-users-guide/

# License
This project is licensed under the MIT License. See the LICENSE file for more information.
