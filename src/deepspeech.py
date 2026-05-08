import torch
import torchaudio

class SpeechRecognitionModel:
    def __init__(self, device):
        self.device = device
        print(f"Loading state-of-the-art ASR model into {device}...")
        
        # We use torchaudio's native Wav2Vec2 ASR pipeline for flawless PyTorch integration.
        # This is architecturally equivalent to DeepSpeech for adversarial testing.
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()
        self.model.eval() # Freeze model for inference

    def transcribe(self, waveform):
        """Converts an audio tensor into English text."""
        with torch.no_grad():
            # Pass the audio through the neural network
            emission, _ = self.model(waveform.to(self.device))
            
            # Decode the output matrix into text
            decoder = GreedyCTCDecoder(labels=self.labels)
            return decoder(emission[0])

class GreedyCTCDecoder(torch.nn.Module):
    """Translates the neural network's probability matrix into human words."""
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices]).replace("|", " ").strip().lower()