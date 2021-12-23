clc;clear all;close all;
addpath('E:\Projects\Utilities')
DirPath = 'Pluto'; %'Haifa'
FilesList = dir(DirPath); 

%% Records Configs
cfg.fs = 12.5e3;
cfg.ExpectedBW = 8e3;
cfg.Decimation_factor = 1;

%% sort by date and remove empty bursts
[~,Inds] = sort([FilesList.datenum].');
mintime = 1.5; %[sec]
fs = 12.5e3;
FileSize = [FilesList.bytes];
min_bytes_size = fs*mintime*(32/8)*2;
ValidInds = Inds(FileSize(Inds)>min_bytes_size);
M = length(ValidInds);

%% Extracting Audio Features
%maxtimen note: if we will choose it to be too big then a alot of spectograms will have too
%many zeros at the end and we will save a big file with redundant data
maxtime = 3.2; %[sec] 2.5->248 , 5->606 
NumBands = 64;
%MelSpectrumArray =zeros(M,NumBands,498,'single'); %time value determind hard coded by dommy sample!! freq resolution isn't relevant because it is logaritmic scale
DemodedMat = zeros(M,round(maxtime*fs),'single');
FreqOffsetMat = zeros(M,1);

h = waitbar(0,'Processing and Demodulating signals...');
for k = 1:1:M
    sig = LoadBinFile([DirPath,'\',FilesList(ValidInds(k)).name]);
    [sig_decimation,FreqOffsetMat(k)] = GenericAMPreProcessing(sig,cfg);
    demod = GenericAMdemodulation(sig_decimation);
    demod = PadOrChop(demod,round(maxtime*fs));
    %MelSpectrumArray(k,:,:) = melSpectrogram(demod,fs,'NumBands',NumBands);
    DemodedMat(k,:) = demod;
    waitbar(k / M)
end
close(h)
%save('HaifaDemoded.mat','DemodedMat');
%save('PlutoDemoded.mat','DemodedMat');

%% TSNE
% Y = tsne(abs(SpectrumMat));
% scatter(Y(:,1),Y(:,2),'.');

%% Useless scatter plot
scatter(FreqOffsetMat,[FilesList(ValidInds).datenum])

%% sample some converstion
rand_ind = randi(M);
for k = rand_ind:rand_ind+3
    sig = LoadBinFile([DirPath,'\',FilesList(ValidInds(k)).name]);
    sig_decimation = GenericAMPreProcessing(sig,cfg);
    %soundsc(abs(sig_decimation),cfg.fs/cfg.Decimation_factor)
    demod = GenericAMdemodulation(sig_decimation);
    sound(demod,cfg.fs/cfg.Decimation_factor);
    pause(length(sig_decimation)/(cfg.fs/cfg.Decimation_factor)+1); %Wait until the sound is over + 1 sec
end

%% What can be classified?
% english \ hebrew (both)
% aircraft \ tower (I think I got from haifa only aircraft and from pluto only the tower)
% army \ civilian (only pluto)
% word trigger (both)

%% A one time run:
% load('E:\Projects\Flight\DLCode\HaifaDemoded.mat')
% HaifaDemoded = DemodedMat;
% load('E:\Projects\Flight\DLCode\PlutoDemoded.mat')
% PlutoDemoded = DemodedMat;
% DemodedMat = [HaifaDemoded;PlutoDemoded];
% save('CombinedDemoded.mat','DemodedMat');
