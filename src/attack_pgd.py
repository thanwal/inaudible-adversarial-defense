import torch

def generate_pgd_attack(model, waveform, target_transcript, epsilon=0.05, alpha=0.01, iters=10):
    """
    Projected Gradient Descent (PGD) implementation for Audio Data.
    Based on the NDSS 2024 specifications for inaudible adversarial perturbations.
    """
    # 1. Freeze the DeepSpeech model (we are attacking the data, not training the network)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Prepare the audio tensor to capture gradients
    perturbed_waveform = waveform.clone().detach().to(waveform.device)
    perturbed_waveform.requires_grad = True
    
    print(f"Generating adversarial perturbation... (Target: '{target_transcript}')")
    
    for i in range(iters):
        model.zero_grad()
        
        # 3. Forward pass through the ASR model
        # Note: In a full pipeline, this feeds into a CTC Loss function 
        # compared against the attacker's target_transcript
        outputs = model(perturbed_waveform)
        
        # Dummy loss for structural scaffolding (will be replaced by actual CTC loss in the pipeline)
        loss = outputs.sum() 
        
        # 4. Backward pass to find the gradients of the AUDIO, not the weights
        loss.backward()
        
        with torch.no_grad():
            # 5. Apply the fast gradient sign method (FGSM) step
            perturbation = alpha * perturbed_waveform.grad.sign()
            perturbed_waveform = perturbed_waveform + perturbation
            
            # 6. Projection: Ensure the noise doesn't exceed the epsilon threshold
            # This keeps the attack "inaudible" to human ears
            noise = torch.clamp(perturbed_waveform - waveform, min=-epsilon, max=epsilon)
            perturbed_waveform = torch.clamp(waveform + noise, min=-1.0, max=1.0)
            
        perturbed_waveform.requires_grad = True
        
    print("Adversarial waveform generated successfully.")
    return perturbed_waveform.detach()