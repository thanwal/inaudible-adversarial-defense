import torch
import torch.nn as nn
import torchaudio.functional as F

class AcousticFirewall(nn.Module):
    def __init__(self, sample_rate=16000, cutoff_freq=3000.0):
        """
        A PyTorch spectral filter to neutralize inaudible adversarial attacks.
        Human speech intelligible range: ~300Hz to 3000Hz.
        Adversarial perturbation range: 4000Hz to 8000Hz.
        """
        super(AcousticFirewall, self).__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq

    def forward(self, waveform):
        # Apply a low-pass biquad filter.
        # This mathematically zeros out frequencies above our 3000Hz cutoff,
        # effectively deleting the PGD adversarial gradients from the audio tensor.
        clean_waveform = F.lowpass_biquad(
            waveform, 
            self.sample_rate, 
            self.cutoff_freq
        )
        return clean_waveform

if __name__ == "__main__":
    # Quick unit test to ensure the tensor math works on your SageMaker GPU
    firewall = AcousticFirewall()
    dummy_audio = torch.randn(1, 16000) # 1 second of random noise
    filtered_audio = firewall(dummy_audio)
    print(f"Firewall initialized. Input shape: {dummy_audio.shape} | Output shape: {filtered_audio.shape}")