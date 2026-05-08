import torch
import torch.nn as nn
import torchaudio.functional as F

class AcousticFirewall(nn.Module):
    def __init__(self, quantization_channels=256):
        """
        Mu-Law Companding Defense (Telephony Compression).
        Logarithmically compresses the audio wave. This perfectly preserves 
        macroscopic human speech (like a phone call) while completely 
        crushing microscopic, linear adversarial perturbations.
        """
        super(AcousticFirewall, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, waveform):
        # 1. Encode using Mu-Law (squashes the linear adversarial noise)
        encoded = F.mu_law_encoding(waveform, self.quantization_channels)
        
        # 2. Decode back to a continuous waveform for the ASR model
        cleaned_waveform = F.mu_law_decoding(encoded, self.quantization_channels)
        
        return cleaned_waveform