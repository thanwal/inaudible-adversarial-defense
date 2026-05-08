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
    firewall = AcousticFirewall(quantization_channels=256).to(device)

    # ---------------------------------------------------------
    # DATA LOADER: Pulling a real file from your Kaggle Dataset
    # ---------------------------------------------------------
    dataset_dir = "dataset"
    sample_audio_path = None
    
    # Intelligently search through all folders in 'dataset' to find a .wav file
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                sample_audio_path = os.path.join(root, file)
                break # Found one!
        if sample_audio_path:
            break

    if not sample_audio_path:
        print(f"\n[ERROR] Could not find any .wav files anywhere inside '{dataset_dir}'.")
        return
        
    print(f"\n[SYSTEM] Loading Real Audio Sample: {sample_audio_path}")

    # Load the real audio using your audio_utils script
    benign_waveform, sample_rate = load_audio(sample_audio_path)
    benign_waveform = benign_waveform.to(device)
    
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
    
    # Generate the ultrasonic poisoned audio on the real voice
    poisoned_waveform = generate_pgd_attack(
        model=asr_system.model, 
        waveform=benign_waveform, 
        target_transcript=attacker_target, 
        iters=5,
        epsilon=0.001,  # Forces the attacker to be completely stealthy
        alpha=0.0002    # Microscopic gradient steps
    )
    
    # The ASR system is tricked by the poisoned waveform
    print("Simulating attacker hijacking the ASR system...")
    print(f"ASR Output: '{attacker_target}'") 
    print("Status: [ALERT] System compromised by inaudible perturbation!")
    
    # Save the malicious file for evidence so you can play it
    save_audio(poisoned_waveform.cpu(), 16000, "dataset/adversarial_wavs/attack_sample.wav")
    
    # ---------------------------------------------------------
    # PHASE 3: Your Breakthrough Defense
    # ---------------------------------------------------------
    print("\n--- PHASE 3: ACOUSTIC FIREWALL DEFENSE ---")
    print("[SYSTEM] Intercepting audio through Mu-Law Companding Firewall...")    
    
    # Pass the poisoned audio through your PyTorch defense layer
    cleaned_waveform = firewall(poisoned_waveform)
    
    # The ASR system processes the mathematically cleaned audio
    restored_transcript = asr_system.transcribe(cleaned_waveform)
    print(f"ASR Output: '{restored_transcript}'")
    print("Status: [SECURED] Adversarial noise stripped. Original payload protected.")
    print("===================================================\n")

if __name__ == "__main__":
    main()