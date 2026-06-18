import torch
import sys
import os
import time
import hashlib

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.lsra.dynamic_multi_agent_core import DynamicMultiAgentCore
from Code.Enterprise.lsra.omni_unified_decoder import NativeOmniDecoder

def type_effect(text, delay=0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def text_to_qpls(prompt, d_model=256):
    """
    Real deterministic conversion from prompt to QPLS using SHA256.
    Ensures orthogonal sparsity by mapping only to the Entity partition [0:64].
    """
    h = hashlib.sha256(prompt.encode()).digest()
    vec = torch.zeros(1, 1, d_model)
    segment_dim = d_model // 4
    for i in range(segment_dim):
        vec[0, 0, i] = (h[i % len(h)] / 128.0) - 1.0 
    return vec

def main():
    print("\n" + "="*60)
    print(" 🧠 Miiri : NATIVE OMNI-CHAT (REAL INFERENCE, ZERO MOCKS)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Booting on {device}...")
    
    # Initialize the pure native architecture matching the master training script exactly
    # Initial experts must match what was saved during training (10 experts after dynamic expansion)
    core = DynamicMultiAgentCore(d_model=256, initial_experts=10).to(device)
    # The decoder vocab sizes must match the training script exactly
    decoder = NativeOmniDecoder(d_model=256, vocab_text=200000, vocab_vision=50000, vocab_audio=10000).to(device)
    
    print("[SYSTEM] Loading Native Unified Weights...")
    try:
        checkpoint = torch.load("Dist/Miiri_Master_Model.pt", map_location=device)
        # strict=False allows the DynamicMultiAgentCore to load despite dynamic growth during training
        core.load_state_dict(checkpoint['reasoning_core'], strict=False)
        decoder.load_state_dict(checkpoint['omni_decoder'])
        print("[OK] Master Weights loaded. Real forward passes enabled.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load Master weights: {e}")
        print("[FATAL ERROR] No fallback allowed (No Mocking Rule). Exiting.")
        sys.exit(1)
        
    print("\n[READY] The architecture runs 100% real tensor operations. No mocks.")
    print("-> Type your message (or 'exit' to quit):")
    
    while True:
        try:
            prompt = input("\nYou > ")
            if prompt.lower() in ['exit', 'quit']:
                break
                
            print("\n[Port 26401] Converting Prompt to QPLS Mentalese Vector (Deterministic Hash)...")
            start_time = time.time()
            input_tensor = text_to_qpls(prompt).to(device)
            
            print(f"[Port 26415-18] Executing TRUE Latent Multi-Agent Debate...")
            with torch.no_grad():
                # DynamicMultiAgentCore takes input_tensor and returns the consensus vector
                consensus_vector = core(input_tensor.squeeze(1))
                
            elapsed_reasoning = time.time() - start_time
            print(f"  -> Consensus reached across {core.num_experts} sub-agents in {elapsed_reasoning:.3f}s.")
            
            print("[Port 26420] Omni-Decoding (Real tensor-to-logit projection)...")
            start_decode = time.time()
            with torch.no_grad():
                logits = decoder(consensus_vector)
                token_id = torch.argmax(logits, dim=-1).item()
                prob = torch.max(torch.softmax(logits, dim=-1)).item()
            elapsed_decode = time.time() - start_decode
            
            if token_id < 200000:
                modality = "TEXTE"
            elif token_id < 250000:
                modality = "IMAGE"
            else:
                modality = "AUDIO"

            type_effect(f"Miiri > [RÉSULTAT MATHÉMATIQUE BRUT] Modality: {modality} | Token ID: {token_id} | Probability: {prob:.4f}")
            print(f"  -> Decoding latency: {elapsed_decode:.4f}s")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
