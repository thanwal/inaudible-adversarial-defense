import torch
import torch.nn as nn

class AcousticFirewall(nn.Module):
    def __init__(self, noise_std=0.02):
        """
        Randomized Smoothing Defense (Gaussian Noise Injection).
        Adversarial attacks rely on mathematically perfect, microscopic gradients. 
        By injecting random Gaussian noise, we 'blind' the attack's precision 
        while the neural network easily hears the human voice through the static.
        """
        super(AcousticFirewall, self).__init__()
        self.noise_std = noise_std

    def forward(self, waveform):
        # Generate random static noise with the exact same shape as the audio
        noise = torch.randn_like(waveform) * self.noise_std
        
        # Add the static to the audio to shatter the PGD attack
        smoothed_waveform = waveform + noise
        
        # Ensure the waveform doesn't exceed physical audio boundaries [-1.0, 1.0]
        return torch.clamp(smoothed_waveform, min=-1.0, max=1.0)