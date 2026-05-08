import torch
import torch.nn as nn

class AcousticFirewall(nn.Module):
    def __init__(self, quantization_steps=256):
        """
        Acoustic Feature Squeezing via Amplitude Quantization.
        Destroys microscopic adversarial gradients by rounding waveform tensors,
        preserving the macroscopic human voice profile perfectly.
        """
        super(AcousticFirewall, self).__init__()
        self.quantization_steps = quantization_steps

    def forward(self, waveform):
        # Multiply, round to the nearest integer step, and divide back.
        # This acts as a mathematical step-function that shatters the fragile, 
        # highly-specific PGD noise while leaving the core human voice intact.
        squeezed_waveform = torch.round(waveform * self.quantization_steps) / self.quantization_steps
        
        return squeezed_waveform