#!/usr/bin/env python
# coding: utf-8

### Import libraries ###
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
import os
import sys
import time
import glob
import gc
import h5py
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
from pystoi import stoi as STOI
from pypesq import pesq as PESQ

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

### Function for pre-processing ###
def pre_processing(data, Fs, down_sample):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Downsample if necessary
    if down_sample is not None:
        wavdata = sg.resample_poly(wavdata, down_sample, Fs)
        Fs = down_sample
    
    return wavdata, Fs

### Function for calculating STFT-Spectrogram ###
def get_STFT(folder_path, down_sample, frame_length, frame_shift, num_frames):
    
    #Initialize list
    x = []
    
    #Get .wav files as an object
    files = sorted(glob.glob(folder_path + "/*.wav"))
    print("Folder:" + folder_path)
    
    #For a progress bar
    nfiles = len(files)
    unit = max(1, math.floor(nfiles/20))
    bar = "#" + " " * math.floor(nfiles/unit)
    
    #Repeat every file-name
    for i, file in enumerate(files):
        
        #Display a progress bar
        print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        if i % unit == 0:
            bar = "#" * math.ceil(i/unit) + " " * math.floor((nfiles-i)/unit)
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        
        #Read .wav file and get pre-process
        wavdata, Fs = sf.read(file)
        wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
        
        #Calculate the index of window size and overlap
        FL = round(frame_length * Fs)
        FS = round(frame_shift * Fs)
        OL = FL - FS
        
        #Execute STFT
        _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
        dft = dft[:-1].T #Remove the last point and get transpose
        spec = np.log10(np.abs(dft))
        
        #Crop the temporal frames into input size
        num_seg = math.floor(spec.shape[0] / num_frames)
        for j in range(num_seg):
            #Add results to list sequentially
            x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
    
    #Finish the progress bar
    bar = "#" * math.ceil(nfiles/unit)
    print("\rProgress:[{0}] {1}/{2} Completed!   ".format(bar, i+1, nfiles), end="")
    print()
    
    #Convert into numpy array
    x = np.array(x)
    
    #Return the result
    return x, Fs

### Function for reading an audio and getting the STFT ###
def read_evaldata(file_path, down_sample, frame_length, frame_shift, num_frames):
    
    #Initialize list
    x = []
    ang_x = []
    
    #Read .wav file and get pre-process
    wavdata, Fs = sf.read(file_path)
    wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    dft = dft[:-1].T #Remove the last point and get transpose
    ang = np.angle(dft) #Preserve the phase
    spec = np.log10(np.abs(dft))
    
    #Crop the temporal frames into input size
    num_seg = math.floor(spec.shape[0] / num_frames)
    for j in range(num_seg):
        #Add results to list sequentially
        x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
        ang_x.append(ang[int(j*num_frames) : int((j+1)*num_frames), :])
    
    #Convert into numpy array
    x = np.array(x)
    ang_x = np.array(ang_x)
    
    return wavdata, Fs, x, ang_x

### Function for reconstructing a waveform ###
def reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift):
    
    #Construct the spectrogram by concatenating all segments
    Y = np.reshape(eval_y, (-1, eval_y.shape[-1]))
    ang = np.reshape(ang_x, (-1, ang_x.shape[-1]))
    
    #The Y and arg can be transpose for processing
    Y, ang = Y.T, ang.T
    
    #Restore the magnitude of STFT
    Y = np.power(10, Y)
    
    #Retrieve the phase from original wave
    Y = Y * np.exp(1j*ang)
    
    #Add the last frequency bin along with frequency axis
    Y = np.append(Y, Y[-1, :][np.newaxis,:], axis=0)
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Custom Dataset class for PyTorch ###
class SpectrogramDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.x_data)
        
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

### U-Net model definition in PyTorch ###
# class UNetModel(nn.Module):
#     def __init__(self):
#         super(UNetModel, self).__init__()
        
#         # Encoder
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 2), padding='same'),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 2), padding='same'),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=(5, 7), stride=(1, 2), padding='same'),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 2), padding='same'),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc5 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(2, 2), padding='same'),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding='same'),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding='same'),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc8 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding='same'),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         # Decoder
#         self.dec1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.dec2 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.dec3 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.dec4 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
        
#         self.dec5 = nn.Sequential(
#             nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(1, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
        
#         self.dec6 = nn.Sequential(
#             nn.ConvTranspose2d(256, 64, kernel_size=(5, 7), stride=(1, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
        
#         self.dec7 = nn.Sequential(
#             nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=0, output_padding=0),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
        
#         self.dec8 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=0, output_padding=0),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)
#         e6 = self.enc6(e5)
#         e7 = self.enc7(e6)
#         e8 = self.enc8(e7)
        
#         # Decoder with skip connections
#         # Note: You'll need to adjust padding and dimensions based on your input size
#         d1 = self.dec1(e8)
#         # Adjust d1 size to match e7 if needed using interpolate
#         d1 = F.interpolate(d1, size=e7.size()[2:], mode='nearest')  
#         d2 = self.dec2(torch.cat([d1, e7], dim=1))
#         d2 = F.interpolate(d2, size=e6.size()[2:], mode='nearest')
#         d3 = self.dec3(torch.cat([d2, e6], dim=1))
#         d3 = F.interpolate(d3, size=e5.size()[2:], mode='nearest')
#         d4 = self.dec4(torch.cat([d3, e5], dim=1))
#         d4 = F.interpolate(d4, size=e4.size()[2:], mode='nearest')
#         d5 = self.dec5(torch.cat([d4, e4], dim=1))
#         d5 = F.interpolate(d5, size=e3.size()[2:], mode='nearest')
#         d6 = self.dec6(torch.cat([d5, e3], dim=1))
#         d6 = F.interpolate(d6, size=e2.size()[2:], mode='nearest')
#         d7 = self.dec7(torch.cat([d6, e2], dim=1))
#         d7 = F.interpolate(d7, size=e1.size()[2:], mode='nearest')
#         d8 = self.dec8(torch.cat([d7, e1], dim=1))
#         d8 = F.interpolate(d8, size=x.size()[2:], mode='nearest')
        
#         return d8

# ### U-Net model definition in PyTorch ###
# class UNetModel(nn.Module):
#     def __init__(self):
#         super(UNetModel, self).__init__()
        
#         # Encoder
#         # Calculate padding for 'same' behavior with strides
#         # For kernel (5,7) with stride (1,2)
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2)),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc5 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc8 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2)
#         )
        
#         # Decoder with padding calculated for each layer
#         self.dec1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.dec2 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.dec3 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.dec4 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
        
#         self.dec5 = nn.Sequential(
#             nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2), output_padding=(0, 1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
        
#         self.dec6 = nn.Sequential(
#             nn.ConvTranspose2d(256, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
        
#         self.dec7 = nn.Sequential(
#             nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
        
#         self.dec8 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)
#         e6 = self.enc6(e5)
#         e7 = self.enc7(e6)
#         e8 = self.enc8(e7)
        
#         # Decoder with skip connections
#         d1 = self.dec1(e8)
#         # The output_padding should handle size matching now, but if there are still issues:
#         if d1.size() != e7.size():
#             d1 = F.interpolate(d1, size=e7.size()[2:], mode='nearest')
            
#         d2 = self.dec2(torch.cat([d1, e7], dim=1))
#         if d2.size() != e6.size():
#             d2 = F.interpolate(d2, size=e6.size()[2:], mode='nearest')
            
#         d3 = self.dec3(torch.cat([d2, e6], dim=1))
#         if d3.size() != e5.size():
#             d3 = F.interpolate(d3, size=e5.size()[2:], mode='nearest')
            
#         d4 = self.dec4(torch.cat([d3, e5], dim=1))
#         if d4.size() != e4.size():
#             d4 = F.interpolate(d4, size=e4.size()[2:], mode='nearest')
            
#         d5 = self.dec5(torch.cat([d4, e4], dim=1))
#         if d5.size() != e3.size():
#             d5 = F.interpolate(d5, size=e3.size()[2:], mode='nearest')
            
#         d6 = self.dec6(torch.cat([d5, e3], dim=1))
#         if d6.size() != e2.size():
#             d6 = F.interpolate(d6, size=e2.size()[2:], mode='nearest')
            
#         d7 = self.dec7(torch.cat([d6, e2], dim=1))
#         if d7.size() != e1.size():
#             d7 = F.interpolate(d7, size=e1.size()[2:], mode='nearest')
            
#         d8 = self.dec8(torch.cat([d7, e1], dim=1))
#         if d8.size() != x.size():
#             d8 = F.interpolate(d8, size=x.size()[2:], mode='nearest')
        
#         return d8

### U-Net model definition in PyTorch ###
class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        
        # Simplified encoder (just 3 layers)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Simplified decoder (just 3 layers)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder with skip connections
        d1 = self.dec1(e3)
        d1 = F.interpolate(d1, size=e2.size()[2:], mode='nearest')
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        d2 = F.interpolate(d2, size=e1.size()[2:], mode='nearest')
        d3 = self.dec3(torch.cat([d2, e1], dim=1))
        d3 = F.interpolate(d3, size=x.size()[2:], mode='nearest')
        
        return d3
    
### Custom loss function (LSD: log-spectral distance) ###
class LSDLoss(nn.Module):
    def __init__(self):
        super(LSDLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # Calculate squared difference
        squared_diff = torch.mean((y_true - y_pred) ** 2, dim=3)
        # Calculate LSD
        lsd = torch.mean(torch.sqrt(squared_diff), dim=2)
        # Return mean LSD
        return torch.mean(lsd)

### Function for executing CNN learning ###
def UNet_learning(train_x, train_y, test_x, test_y, LR, BS, EP):
    # Create dataset and dataloader
    train_dataset = SpectrogramDataset(train_x, train_y)
    test_dataset = SpectrogramDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)
    
    # Initialize model
    model = UNetModel().to(device)
    
    # Print model summary
    print(model)
    
    # Define loss function and optimizer
    criterion = LSDLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9))
    
    # For learning rate decay
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
                                           lambda epoch: 10**(-lr_decay*epoch))
    
    # Lists to store loss values
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(EP):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{EP}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Save model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/unet_model.pth')
    
    # Save loss history
    os.makedirs('./log', exist_ok=True)
    with open('./log/loss_function.txt', 'w') as fp:
        fp.write("epoch\tloss\tval_loss\n")
        for i in range(len(train_losses)):
            fp.write(f"{i}\t{train_losses[i]}\t{val_losses[i]}\n")
    
    # Plot loss
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(18, 5))
    plt.plot(train_losses, label="loss for training")
    plt.plot(val_losses, label="loss for validation")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("./log/loss_function.png", format="png", dpi=300)
    plt.show()
    
    return model

### Function for evaluation ###
# def UNet_evaluation(eval_x):
#     # Create model
#     model = UNetModel().to(device)
    
#     # Load model weights
#     model.load_state_dict(torch.load('./models/unet_model.pth'))
#     model.eval()
    
#     # Convert input to tensor
#     eval_tensor = torch.tensor(eval_x, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    
#     # Predict
#     with torch.no_grad():
#         output = model(eval_tensor)
    
#     # Convert back to numpy
#     eval_y = output.squeeze(1).cpu().numpy()
    
#     return eval_y

### Function for evaluation with RTF calculation ###
def UNet_evaluation(eval_x):
    # Create model
    model = UNetModel().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model parameters: {total_params:,}')
    
    # Try to load model weights
    try:
        model.load_state_dict(torch.load('./models/unet_model.pth'))
        print("Loaded model from './models/unet_model.pth'")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try to find checkpoints
        checkpoints = sorted(glob.glob('./models/unet_checkpoint_*.pth'))
        if checkpoints:
            print(f"Loading latest checkpoint: {checkpoints[-1]}")
            model.load_state_dict(torch.load(checkpoints[-1]))
        else:
            print("No model or checkpoint found. Please train the model first.")
            return None
    
    model.eval()
    
    # For RTF calculation
    total_audio_duration = eval_x.shape[0] * num_frames * frame_shift
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_batches = math.ceil(eval_x.shape[0] / batch_size)
    eval_y = np.zeros_like(eval_x)
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, eval_x.shape[0])
            
            # Convert batch to tensor
            batch_x = torch.tensor(eval_x[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
            
            # Predict
            output = model(batch_x)
            
            # Convert back to numpy
            eval_y[start_idx:end_idx] = output.squeeze(1).cpu().numpy()
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate RTF
    rtf = processing_time / total_audio_duration
    print(f"\nReal-Time Factor (RTF): {rtf:.4f}")
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")
    print(f"Total processing time: {processing_time:.2f} seconds")
    
    # Save RTF to log
    os.makedirs('./log', exist_ok=True)
    with open("./log/rtf_log.txt", "w") as f:
        f.write(f"Real-Time Factor (RTF): {rtf:.4f}\n")
        f.write(f"Total audio duration: {total_audio_duration:.2f} seconds\n")
        f.write(f"Total processing time: {processing_time:.2f} seconds\n")
        f.write(f"Total model parameters: {total_params:,}\n")
    
    return eval_y, rtf

### Main ###
# if __name__ == "__main__":
    
#     #Set up
#     down_sample = 16000    #Downsampling rate (Hz) [Default]16000
#     frame_length = 0.032   #STFT window width (second) [Default]0.032
#     frame_shift = 0.016    #STFT window shift (second) [Default]0.016
#     num_frames = 16        #The number of frames for an input [Default]16
#     learn_rate = 1e-4      #Learning rate for CNN training [Default]1e-4
#     lr_decay = 0           #Learning rate is according to "learn_rate*10**(-lr_decay*n_epoch)" [Default]0
#     batch_size = 8        #Size of batch for CNN training [Default]64
#     epoch = 30             #The number of iteration for CNN training [Default]30
#     mode = "train"         #Select either "train" or "eval" [Default]"train"
#     stft = False            #True: compute STFT from the beginning, False: read numpy files [Default]True
#     num_sample = 800        #The number of samples for evaluation [Default]800
    
#     #For training
#     if mode == "train":
        
#         #In case of computing the STFT at the beginning
#         if stft == True:
#             #Compute STFT for the mixed source
#             fpath = "./audio_data/training/NOISY"
#             train_x, Fs = get_STFT(fpath, down_sample, frame_length, frame_shift, num_frames)
            
#             #Compute STFT for the separated source
#             fpath = "./audio_data/training/CLEAN"
#             train_y, Fs = get_STFT(fpath, down_sample, frame_length, frame_shift, num_frames)
            
#             #Save the training data
#             os.makedirs('./numpy_files', exist_ok=True)
#             np.save('./numpy_files/X_train', train_x)
#             np.save('./numpy_files/Y_train', train_y)
        
#         #In case of reading the STFT spectrogram from local file
#         else:
#             #Read the training data
#             fpath = "./numpy_files"
#             train_x = np.load(fpath + '/X_train.npy')
#             train_y = np.load(fpath + '/Y_train.npy')
        
#         #Remove segments including -inf entries in train_x
#         idx = np.unique(np.where(train_x == -np.inf)[0])
#         idx = list(set(range(train_x.shape[0])) - set(idx)) #Remove indices of -inf
#         train_x = train_x[idx, :, :]
#         train_y = train_y[idx, :, :]
        
#         #Remove segments including -inf entries in train_y
#         idx = np.unique(np.where(train_y == -np.inf)[0])
#         idx = list(set(range(train_y.shape[0])) - set(idx)) #Remove indices of -inf
#         train_x = train_x[idx, :, :]
#         train_y = train_y[idx, :, :]
        
#         #Split the input data into a training set and a small test set
#         test_x = train_x[:3000, :, :]
#         test_y = train_y[:3000, :, :]
#         train_x = train_x[3000:, :, :]
#         train_y = train_y[3000:, :, :]
        
#         #Normalization (not standardization)
#         max_x = np.amax(train_x, axis=None)
#         min_x = np.amin(train_x, axis=None)
#         print(max_x)
#         print(min_x)
#         train_x = (train_x - min_x) / (max_x - min_x)
#         train_y = (train_y - min_x) / (max_x - min_x)
#         test_x = (test_x - min_x) / (max_x - min_x)
#         test_y = (test_y - min_x) / (max_x - min_x)
        
#         #Call my function for executing CNN learning
#         UNet_learning(train_x, train_y, test_x, test_y, learn_rate, batch_size, epoch)
#         print("Done")
    
#     #For evaluation
#     elif mode == "eval":
        
#         #Compute STFT for the mixed source
#         fpath = "./audio_data/evaluation/NOISY"
#         print("Folder:" + fpath)
        
#         #Get .wav files as an object
#         files = sorted(glob.glob(fpath + "/*.wav"))
#         samples = random.sample(list(range(len(files))), k=min(num_sample, len(files))) #Extract samples randomly
        
#         #Define valuables for metrics
#         nfiles = len(samples)
#         PESQ_mix, STOI_mix, ESTOI_mix = np.zeros(nfiles), np.zeros(nfiles), np.zeros(nfiles)
#         PESQ_sep, STOI_sep, ESTOI_sep = np.zeros(nfiles), np.zeros(nfiles), np.zeros(nfiles)
        
#         #For a progress bar
#         if nfiles >= 20:
#             unit = math.floor(nfiles/20)
#         else:
#             unit = math.floor(nfiles/1)
#         bar = "#" + " " * math.floor(nfiles/unit)
        
#         #Repeat for each file
#         for i, sample in enumerate(samples):
            
#             #Display a progress bar
#             print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
#             if i % unit == 0:
#                 bar = "#" * math.ceil(i/unit) + " " * math.floor((nfiles-i)/unit)
#                 print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
            
#             #Call my function for reading audio
#             mix_wav, Fs, eval_x, ang_x = read_evaldata(files[sample], down_sample, frame_length, frame_shift, num_frames)
            
#             #Normalization
#             max_x = -0.2012536819610933 #From training step
#             min_x = -9.188868273733446 #From training step
#             eval_x = (eval_x - min_x) / (max_x - min_x)
            
#             #Call my function for separating audio by the pre-learned U-Net model
#             eval_y = UNet_evaluation(eval_x)
            
#             #Restore the scale before normalization
#             eval_y = eval_y * (max_x - min_x) + min_x
            
#             #Call my function for reconstructing the waveform
#             sep_wav, Fs = reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift)
            
#             #Read the ground truth
#             CLEAN_path = files[sample].replace(fpath.split("/")[-1], 'CLEAN')
#             clean_wav, Fs = sf.read(CLEAN_path)
#             clean_wav, Fs = pre_processing(clean_wav, Fs, down_sample)
            
#             #Adjust the length of audio
#             diff = int(mix_wav.shape[0]) - int(sep_wav.shape[0])
#             if diff > 0:
#                 mix_wav = mix_wav[:-diff]
#                 clean_wav = clean_wav[:-diff]
#             else:
#                 sep_wav = sep_wav[:diff]
            
#             #Compute the PESQ and STOI scores
#             PESQ_mix[i] = PESQ(Fs, clean_wav, mix_wav, 'wb')
#             STOI_mix[i] = STOI(clean_wav, mix_wav, Fs, extended=False)
#             ESTOI_mix[i] = STOI(clean_wav, mix_wav, Fs, extended=True)
#             PESQ_sep[i] = PESQ(Fs, clean_wav, sep_wav, 'wb')
#             STOI_sep[i] = STOI(clean_wav, sep_wav, Fs, extended=False)
#             ESTOI_sep[i] = STOI(clean_wav, sep_wav, Fs, extended=True)
        
#         #Finish the progress bar
#         bar = "#" * math.ceil(nfiles/unit)
#         print("\rProgress:[{0}] {1}/{2} Completed!   ".format(bar, i+1, nfiles), end="")
#         print()
        
#         #Compute the average scores
#         avePESQ_mix, aveSTOI_mix, aveESTOI_mix = np.mean(PESQ_mix), np.mean(STOI_mix), np.mean(ESTOI_mix)
#         avePESQ_sep, aveSTOI_sep, aveESTOI_sep = np.mean(PESQ_sep), np.mean(STOI_sep), np.mean(ESTOI_sep)
#         print("PESQ(original): {:.4f}, STOI(original): {:.4f}, ESTOI(original): {:.4f}".format(avePESQ_mix, aveSTOI_mix, aveESTOI_mix))
#         print("PESQ(separated): {:.4f}, STOI(separated): {:.4f}, ESTOI(separated): {:.4f}".format(avePESQ_sep, aveSTOI_sep, aveESTOI_sep))
        
#         #Save as a log file
#         os.makedirs('./log', exist_ok=True)
#         with open("./log/evaluation.txt", "w") as f:
#             f.write("PESQ(original): {:.4f}, STOI(original): {:.4f}, ESTOI(original): {:.4f}\n".format(avePESQ_mix, aveSTOI_mix, aveESTOI_mix))
#             f.write("PESQ(separated): {:.4f}, STOI(separated): {:.4f}, ESTOI(separated): {:.4f}\n".format(avePESQ_sep, aveSTOI_sep, aveESTOI_sep))

### Main ###
if __name__ == "__main__":
    
    #Set up
    down_sample = 16000    #Downsampling rate (Hz) [Default]16000
    frame_length = 0.032   #STFT window width (second) [Default]0.032
    frame_shift = 0.016    #STFT window shift (second) [Default]0.016
    num_frames = 16        #The number of frames for an input [Default]16
    learn_rate = 1e-4      #Learning rate for CNN training [Default]1e-4
    lr_decay = 0           #Learning rate is according to "learn_rate*10**(-lr_decay*n_epoch)" [Default]0
    batch_size = 16        #REDUCED batch size (from 64 to 16)
    epoch = 3              #REDUCED epochs (from 30 to 3)
    mode = "train"         #Select either "train" or "eval" [Default]"train"
    stft = False           #Use saved STFT if available
    num_sample = 10        #REDUCED samples for evaluation (from 800 to 10)
    
    #For training
    if mode == "train":
        
        #In case of computing the STFT at the beginning
        if stft == True:
            #Compute STFT for the mixed source
            fpath = "./audio_data/training/NOISY"
            train_x, Fs = get_STFT(fpath, down_sample, frame_length, frame_shift, num_frames)
            
            #Compute STFT for the separated source
            fpath = "./audio_data/training/CLEAN"
            train_y, Fs = get_STFT(fpath, down_sample, frame_length, frame_shift, num_frames)
            
            #Save the training data
            os.makedirs('./numpy_files', exist_ok=True)
            np.save('./numpy_files/X_train', train_x)
            np.save('./numpy_files/Y_train', train_y)
        
        #In case of reading the STFT spectrogram from local file
        else:
            #Read the training data
            fpath = "./numpy_files"
            train_x = np.load(fpath + '/X_train.npy')
            train_y = np.load(fpath + '/Y_train.npy')
        
        # REDUCE DATASET SIZE DRASTICALLY for proof of concept
        train_x = train_x[:500]  # Just use 500 samples
        train_y = train_y[:500]
        
        #Remove segments including -inf entries in train_x
        idx = np.unique(np.where(train_x == -np.inf)[0])
        idx = list(set(range(train_x.shape[0])) - set(idx)) #Remove indices of -inf
        train_x = train_x[idx, :, :]
        train_y = train_y[idx, :, :]
        
        #Remove segments including -inf entries in train_y
        idx = np.unique(np.where(train_y == -np.inf)[0])
        idx = list(set(range(train_y.shape[0])) - set(idx)) #Remove indices of -inf
        train_x = train_x[idx, :, :]
        train_y = train_y[idx, :, :]
        
        #Split the input data into a training set and a small test set
        test_x = train_x[:100, :, :]  # Use just 100 samples for testing
        test_y = train_y[:100, :, :]
        train_x = train_x[100:, :, :]
        train_y = train_y[100:, :, :]
        
        print(f"Training with reduced dataset: {train_x.shape}")
        print(f"Testing with reduced dataset: {test_x.shape}")
        
        #Normalization (not standardization)
        max_x = np.amax(train_x, axis=None)
        min_x = np.amin(train_x, axis=None)
        print(max_x)
        print(min_x)
        train_x = (train_x - min_x) / (max_x - min_x)
        train_y = (train_y - min_x) / (max_x - min_x)
        test_x = (test_x - min_x) / (max_x - min_x)
        test_y = (test_y - min_x) / (max_x - min_x)
        
        # Save normalization values for evaluation
        norm_values = {'min': min_x, 'max': max_x}
        np.save('./numpy_files/norm_values.npy', norm_values)
        
        # Add debugging print statements before calling UNet_learning
        print("About to start UNet_learning function")
        
        #Call function for executing CNN learning
        UNet_learning(train_x, train_y, test_x, test_y, learn_rate, batch_size, epoch)
        print("Done with proof of concept training!")