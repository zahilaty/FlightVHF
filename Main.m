clc;clear all;close all;
addpath('D:\Utilities')
DirPath = 'Haifa';
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

%% Features
DFTsize = 1024;
SpectrumMat =zeros(M,DFTsize); %freq resolution is 12.5e3/1024= 12Hz
FreqOffsetMat = zeros(M,1);

for k = 1:1:M
    sig = LoadBinFile([DirPath,'\',FilesList(ValidInds(k)).name]);
    [sig_decimation,FreqOffsetMat(k)] = GenericAMPreProcessing(sig,cfg);
    %SpectrumMat(k,:) = Spectrum_with_2k_size(sig,log2(DFTsize));
end

%% TSNE
% Y = tsne(abs(SpectrumMat));
% scatter(Y(:,1),Y(:,2),'.');

%% 
scatter(FreqOffsetMat,FilesList(ValidInds).datenum)

%% sample some converstion
rand_ind = randi(M);
for k = rand_ind:rand_ind+3
    sig = LoadBinFile([DirPath,'\',FilesList(ValidInds(k)).name]);
    sig_decimation = GenericAMPreProcessing(sig,cfg);
    %soundsc(abs(sig_decimation),cfg.fs/cfg.Decimation_factor)
    demod = GenericAMdemodulation(sig);
    sound(demod,cfg.fs/cfg.Decimation_factor);
    pause(length(sig_decimation)/(cfg.fs/cfg.Decimation_factor)+1);
end




%% What can be classified?
% english \ hebrew
% aircraft \ tower
% army \ civilian?