import torchaudio
import torch
import os

def load_audio(file_path, target_sample_rate=16000):
    """
    Loads a .wav file and forces it to 16kHz so it doesn't crash the DeepSpeech model.
    """
    print(f"Loading {file_path}...")
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample if the audio isn't exactly 16kHz
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        
    # Convert stereo to mono if necessary (DeepSpeech expects 1 channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform, target_sample_rate

def save_audio(waveform, sample_rate, file_path):
    """
    Saves the PyTorch tensor back into a physical .wav file you can listen to.
    """
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    torchaudio.save(file_path, waveform, sample_rate)
    print(f"Audio saved successfully to {file_path}")