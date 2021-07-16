clc;clear all;close all;
addpath('D:\Utilities')
%% load the signal
%file = "D:\Flight\Records_23_3_2021.bin";
file =  "Records_23_3_2021.bin"
fileID = fopen(file,'r');
A = fread(fileID,'single');
fclose(fileID);
sig = A(1:2:end-1) + 1j*A(2:2:end);
N  = length(sig);
clear A

%% load the already demodulate signal to learn how to label right
load demodulate_signal;
Fs = 12500;

%%
figure;plot(db(sig));grid on;xlabel('t[sec]');ylabel('Mag[dB]');title('Signal at time domain')%ylim([-50 -25]);
MRsignal = 10*log10(movmean(abs(sig).^2,[10000 0]));

%% create the plot of audio samples
figure;hold on
plot(MRsignal,'b');title('Signal before demodulation')
xlabel(strcat('Sample Number (fs = ', num2str(Fs), ')'));
ylabel('Magnitude of MA');
ylimits = get(gca, 'YLim'); % get the y-axis limits
plotdata = [ylimits(1):0.1:ylimits(2)];
hline = plot(repmat(0, size(plotdata)), plotdata, 'r'); % plot the marker

%% instantiate the audioplayer object
player = audioplayer(AudioSig, Fs);

%% setup the timer for the audioplayer object
player.TimerFcn = {@plotMarker, player, gcf, plotdata}; % timer callback function (defined below)
player.TimerPeriod = 0.01; % period of the timer in seconds

%% start playing the audio
% this will move the marker over the audio plot at intervals of 0.01 s
play(player);