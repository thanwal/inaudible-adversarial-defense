import torch
import os
import time
from src.audio_utils import load_audio
from src.deepspeech import SpeechRecognitionModel
from src.attack_pgd import generate_pgd_attack
from src.defense_filter import AcousticFirewall

def main():
    print("=== AI Security: Dataset Evaluation Pipeline ===")
    
    # Setup Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_system = SpeechRecognitionModel(device)
    firewall = AcousticFirewall(quantization_channels=256).to(device)
    
    dataset_dir = "dataset/wavs"
    attacker_target = "unlock front door"
    
    # Grab all .wav files (let's limit to 50 for a fast evaluation run)
    # You can increase max_samples for your final paper
    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.wav'):
                all_files.append(os.path.join(root, f))
                
    max_samples = 50
    files_to_test = all_files[:max_samples]
    
    print(f"\n[SYSTEM] Commencing batch evaluation on {len(files_to_test)} audio samples...")
    
    # Tracking Metrics
    attack_success_count = 0
    defense_success_count = 0
    start_time = time.time()

    for idx, audio_path in enumerate(files_to_test):
        print(f"\nProcessing Sample {idx + 1}/{len(files_to_test)}...")
        
        # 1. Load Data
        benign_waveform, _ = load_audio(audio_path)
        benign_waveform = benign_waveform.to(device)
        
        # 2. Execute Attack
        poisoned_waveform = generate_pgd_attack(
            model=asr_system.model, 
            waveform=benign_waveform, 
            target_transcript=attacker_target, 
            iters=5,
            epsilon=0.001,
            alpha=0.0002
        )
        
        # Test Attack Success
        attack_transcript = asr_system.transcribe(poisoned_waveform)
        if attacker_target in attack_transcript:
            attack_success_count += 1
            
        # 3. Execute Defense
        cleaned_waveform = firewall(poisoned_waveform)
        defense_transcript = asr_system.transcribe(cleaned_waveform)
        
        # Test Defense Success (Did it prevent the target phrase?)
        if attacker_target not in defense_transcript:
            defense_success_count += 1

    # --- FINAL REPORT GENERATION ---
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    
    attack_success_rate = (attack_success_count / len(files_to_test)) * 100
    defense_success_rate = (defense_success_count / len(files_to_test)) * 100
    
    print("\n===================================================")
    print("             FINAL EVALUATION REPORT               ")
    print("===================================================")
    print(f"Total Samples Tested:      {len(files_to_test)}")
    print(f"Execution Time:            {execution_time} seconds")
    print("---------------------------------------------------")
    print(f"Attack Success Rate (ASR): {attack_success_rate}%")
    print(f"Firewall Defense Rate:     {defense_success_rate}%")
    print("===================================================")

if __name__ == "__main__":
    # Suppress the load_audio print statements for clean terminal output
    import builtins
    original_print = builtins.print
    def custom_print(*args, **kwargs):
        if "Loading" not in str(args[0]):
            original_print(*args, **kwargs)
    builtins.print = custom_print
    
    main()