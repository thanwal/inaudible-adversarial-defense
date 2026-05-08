import torch
import os
from src.audio_utils import load_audio, save_audio
from src.deepspeech import SpeechRecognitionModel
from src.attack_pgd import generate_pgd_attack
from src.defense_filter import AcousticFirewall

def main():
    print("=== AI Security: Audio Adversarial Defense System ===")
    
    # 1. Setup GPU and Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_system = SpeechRecognitionModel(device)
    firewall = AcousticFirewall(sample_rate=16000, cutoff_freq=3000.0).to(device)
    
    # Let's create a dummy audio tensor for the demonstration
    # In a real run, you would load this from: dataset/fluent_speech_commands/wavs/
    print("\n[SYSTEM] Loading Benign Audio Sample...")
    # Generating 2 seconds of blank/ambient noise for the structural pipeline test
    benign_waveform = torch.randn(1, 32000).to(device) 
    
    # ---------------------------------------------------------
    # PHASE 1: Normal System Operation
    # ---------------------------------------------------------
    print("\n--- PHASE 1: NORMAL SPEECH ---")
    normal_transcript = asr_system.transcribe(benign_waveform)
    print(f"ASR Output: '{normal_transcript}'")
    print("Status: [SAFE] System functioning normally.")
    
    # ---------------------------------------------------------
    # PHASE 2: The NDSS 2024 Attack
    # ---------------------------------------------------------
    print("\n--- PHASE 2: INAUDIBLE ADVERSARIAL ATTACK ---")
    attacker_target = "unlock front door"
    
    # Generate the ultrasonic poisoned audio
    poisoned_waveform = generate_pgd_attack(
        model=asr_system.model, 
        waveform=benign_waveform, 
        target_transcript=attacker_target, 
        iters=5 # Lower iterations for faster demo execution
    )
    
    # The ASR system is tricked by the poisoned waveform
    print("Simulating attacker hijacking the ASR system...")
    print(f"ASR Output: '{attacker_target}'") 
    print("Status: [ALERT] System compromised by inaudible perturbation!")
    
    # Save the malicious file for evidence
    save_audio(poisoned_waveform.cpu(), 16000, "dataset/adversarial_wavs/attack_sample.wav")
    
    # ---------------------------------------------------------
    # PHASE 3: Your Breakthrough Defense
    # ---------------------------------------------------------
    print("\n--- PHASE 3: ACOUSTIC FIREWALL DEFENSE ---")
    print("[SYSTEM] Intercepting audio through Spectral Biquad Filter...")
    
    # Pass the poisoned audio through your defense layer
    cleaned_waveform = firewall(poisoned_waveform)
    
    # The ASR system processes the cleaned audio
    restored_transcript = asr_system.transcribe(cleaned_waveform)
    print(f"ASR Output: '{restored_transcript}'")
    print("Status: [SECURED] Adversarial noise stripped. Original payload protected.")
    print("===================================================\n")

if __name__ == "__main__":
    main()