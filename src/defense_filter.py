import torch
import torch.nn as nn
import torchaudio.transforms as T

class AcousticFirewall(nn.Module):
    def __init__(self, sample_rate=16000, cutoff_freq=None):
        """
        Acoustic Feature Squeezing Defense.
        Shatters fragile adversarial gradients via rapid downsampling/upsampling,
        preserving the core phonetic structures required by the ASR model.
        """
        super(AcousticFirewall, self).__init__()
        
        # We compress the audio to 8000Hz, then immediately restore it to 16000Hz
        self.downsample = T.Resample(orig_freq=sample_rate, new_freq=8000)
        self.upsample = T.Resample(orig_freq=8000, new_freq=sample_rate)

    def forward(self, waveform):
        # 1. Compress the audio to crush the adversarial noise
        squeezed_audio = self.downsample(waveform)
        
        # 2. Decompress back to 16kHz so the ASR model can process it normally
        cleaned_audio = self.upsample(squeezed_audio)
        
        return cleaned_audio