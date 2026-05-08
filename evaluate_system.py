import torch
import os
import time
import shutil
from src.audio_utils import load_audio, save_audio
from src.deepspeech import SpeechRecognitionModel
from src.attack_pgd import generate_pgd_attack
from src.defense_filter import AcousticFirewall

def main():
    # 1. Create a clean Demo folder
    demo_folder = "demo_outputs"
    if os.path.exists(demo_folder):
        shutil.rmtree(demo_folder)
    os.makedirs(demo_folder, exist_ok=True)

    print("=== AI Security: Audio Adversarial Defense System ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_system = SpeechRecognitionModel(device)
    firewall = AcousticFirewall(quantization_channels=256).to(device)

    # ---------------------------------------------------------
    # 2. Find and Save Original Audio
    # ---------------------------------------------------------
    dataset_dir = "dataset"
    sample_audio_path = None
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                sample_audio_path = os.path.join(root, file)
                break 
        if sample_audio_path: break

    print(f"\n[SYSTEM] Loading: {sample_audio_path}")
    benign_waveform, _ = load_audio(sample_audio_path)
    
    # Save a copy to the demo folder
    original_demo_path = os.path.join(demo_folder, "01_original_human_voice.wav")
    save_audio(benign_waveform, 16000, original_demo_path)

    # ---------------------------------------------------------
    # 3. PHASE 1: Normal Operation
    # ---------------------------------------------------------
    print("\n--- PHASE 1: NORMAL SPEECH ---")
    normal_transcript = asr_system.transcribe(benign_waveform.to(device))
    print(f"ASR Output: '{normal_transcript}'")
    
    # ---------------------------------------------------------
    # 4. PHASE 2: Generate & Save Attack
    # ---------------------------------------------------------
    print("\n--- PHASE 2: GENERATING ATTACK ---")
    attacker_target = "unlock front door"
    poisoned_waveform = generate_pgd_attack(
        model=asr_system.model, 
        waveform=benign_waveform.to(device), 
        target_transcript=attacker_target, 
        iters=5, epsilon=0.001, alpha=0.0002    
    )
    
    attack_demo_path = os.path.join(demo_folder, "02_adversarial_attack.wav")
    save_audio(poisoned_waveform.cpu(), 16000, attack_demo_path)
    
    print(f"ASR Output (Attacked): '{attacker_target}'") 

    # ---------------------------------------------------------
    # 5. PHASE 3: Generate & Save Cleaned Audio
    # ---------------------------------------------------------
    print("\n--- PHASE 3: APPLYING FIREWALL ---")    
    cleaned_waveform = firewall(poisoned_waveform)
    
    cleaned_demo_path = os.path.join(demo_folder, "03_cleaned_by_firewall.wav")
    save_audio(cleaned_waveform.cpu(), 16000, cleaned_demo_path)
    
    restored_transcript = asr_system.transcribe(cleaned_waveform)
    print(f"ASR Output (Defended): '{restored_transcript}'")

    print("\n" + "="*50)
    print("✅ DEMO FILES READY!")
    print(f"Go to the folder: '{demo_folder}'")
    print("Right-click and download the 3 files to play them locally.")
    print("="*50)

if __name__ == "__main__":
    main()