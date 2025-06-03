# models/unet/unet_wrapper.py
"""
EXACT wrapper that uses the same preprocessing/postprocessing functions
from U-Net_speech_enhancement.py
"""
import torch
import numpy as np
import soundfile as sf
from scipy import signal as sg
import math
import os
from core.base_model import BaseModel

class UNetWrapper(BaseModel):
    """UNet wrapper with EXACT functions from U-Net_speech_enhancement.py"""
    
    def __init__(self, model_class, config):
        """
        Initialize UNet wrapper exactly like the original
        """
        # Call parent constructor - but we'll override most functionality
        super().__init__(model_class, config)
        
        # EXACT parameters from original
        self.down_sample = 16000
        self.frame_length = 0.032
        self.frame_shift = 0.016
        self.num_frames = 16
        
        # EXACT normalization values from original training
        self.max_x = -0.2012536819610933  # From original training step
        self.min_x = -9.188868273733446   # From original training step
    
    def pre_processing(self, data, Fs, down_sample):
        """EXACT copy of pre_processing function from original"""
        
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
    
    def read_evaldata(self, file_path, down_sample, frame_length, frame_shift, num_frames):
        """EXACT copy of read_evaldata function from original"""
        
        #Initialize list
        x = []
        ang_x = []
        
        #Read .wav file and get pre-process
        wavdata, Fs = sf.read(file_path)
        wavdata, Fs = self.pre_processing(wavdata, Fs, down_sample)
        
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
    
    def reconstruct_wave(self, eval_y, ang_x, Fs, frame_length, frame_shift):
        """EXACT copy of reconstruct_wave function from original"""
        
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
    
    def UNet_evaluation(self, eval_x):
        """EXACT copy of UNet_evaluation function from original (without RTF calculation)"""
        
        # Convert input to tensor
        eval_tensor = torch.tensor(eval_x, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(eval_tensor)
        
        # Convert back to numpy
        eval_y = output.squeeze(1).cpu().numpy()
        
        return eval_y
    
    def preprocess_for_inference(self, audio_path):
        """
        Preprocess audio using EXACT original functions
        """
        # Use the exact read_evaldata function
        wavdata, Fs, eval_x, ang_x = self.read_evaldata(
            audio_path, self.down_sample, self.frame_length, self.frame_shift, self.num_frames
        )
        
        # Apply EXACT normalization from original
        eval_x = (eval_x - self.min_x) / (self.max_x - self.min_x)
        
        return eval_x, {
            'ang_x': ang_x,
            'Fs': Fs,
            'wavdata': wavdata
        }
    
    def postprocess_output(self, model_output, reconstruction_info):
        """
        Postprocess using EXACT original functions
        """
        # Restore the scale before normalization (EXACT from original)
        eval_y = model_output * (self.max_x - self.min_x) + self.min_x
        
        # Use exact reconstruct_wave function
        rec_wav, Fs = self.reconstruct_wave(
            eval_y, 
            reconstruction_info['ang_x'], 
            reconstruction_info['Fs'], 
            self.frame_length, 
            self.frame_shift
        )
        
        return rec_wav
    
    def inference(self, audio_path):
        """
        Complete inference pipeline using EXACT original logic
        """
        # Preprocess using exact functions
        eval_x, reconstruction_info = self.preprocess_for_inference(audio_path)
        
        # Run model inference using exact function
        eval_y = self.UNet_evaluation(eval_x)
        
        # Postprocess using exact functions
        reconstructed_audio = self.postprocess_output(eval_y, reconstruction_info)
        
        return reconstructed_audio