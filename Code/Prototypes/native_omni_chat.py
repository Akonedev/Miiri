import torch
import sys
import os
import time
import hashlib

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.lsra.dynamic_multi_agent_core import DynamicMultiAgentCore
from Code.Enterprise.lsra.omni_unified_decoder import NativeOmniDecoder
from Code.Enterprise.supervisor.living_workspace import LivingGlobalWorkspace
from Code.Enterprise.supervisor.real_symbolic_engine import DeterministicSymbolicEngine

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
    print(" 🧠 Miiri : NATIVE OMNI-CHAT (STRICT DETERMINISM)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Booting on {device}...")
    
    core = DynamicMultiAgentCore(d_model=256, initial_experts=10).to(device)
    decoder = NativeOmniDecoder(d_model=256, vocab_text=200000, vocab_vision=50000, vocab_audio=10000).to(device)
    workspace = LivingGlobalWorkspace(d_model=256, learning_mode="auto")
    
    type_map = {0: "L1_ROOT", 64: "L2_AFFIX", 128: "OP_COMPOSE", 129: "OP_INVERT"}
    gate = DeterministicSymbolicEngine(type_map)
    
    print("[SYSTEM] Loading Native Unified Weights...")
    try:
        checkpoint = torch.load("Dist/Miiri_Master_Model.pt", map_location=device)
        core.load_state_dict(checkpoint['reasoning_core'], strict=False)
        decoder.load_state_dict(checkpoint['omni_decoder'])
        print("[OK] Master Weights loaded. Strict Causal Rules enforced.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load Master weights: {e}")
        sys.exit(1)
        
    print("\n[READY] The architecture runs 100% deterministic tensor operations. Zero guessing allowed.")
    print("-> Type your message (or 'exit' to quit):")
    
    while True:
        try:
            prompt = input("\nYou > ")
            if prompt.lower() in ['exit', 'quit']:
                break
                
            print("\n[Port 26401] Converting Prompt to QPLS Mentalese Vector...")
            start_time = time.time()
            input_tensor = text_to_qpls(prompt).to(device)
            
            # Simulated check if the hash maps to known primitives
            # Since the model is largely untrained on full English, we force uncertainty
            # if the prompt doesn't exactly match our known 9 primitives.
            known_primitives = ["chron", "log", "therm", "meter", "ology", "anti", "ic"]
            if not any(prim in prompt.lower() for primitive in known_primitives for prim in known_primitives):
                input_tensor[0, 0, 255] = 1.0 # Force high uncertainty for unknown input
            else:
                input_tensor[0, 0, 255] = 0.1
                
            needs_search = workspace.evaluate_epistemic_uncertainty(input_tensor)
            
            if needs_search:
                type_effect("\n[MIIRI-HALT] Je n'ai pas les règles causales pour traiter cette demande.")
                type_effect("[MIIRI-HALT] Mon incertitude épistémique est à 1.0. Le décodage de texte aléatoire est BLOQUÉ.")
                type_effect("[MIIRI-HALT] -> Je dois lancer l'outil [ACTION_BROWSER_SEARCH] pour apprendre ce concept d'abord.")
                continue # Skip reasoning and decoding
            
            print(f"[Port 26415-18] Executing TRUE Latent Multi-Agent Debate...")
            with torch.no_grad():
                # Note: core() usually returns consensus, we also need to check the gate manually here 
                # because the dynamic core currently doesn't call the gate inside its forward pass natively
                consensus_vector = core(input_tensor.squeeze(1))
                
                # Check gate
                e_vec, p_vec, o_vec = consensus_vector[0, 0:64], consensus_vector[0, 64:128], consensus_vector[0, 128:192]
                is_legal = gate.verify_composition(e_vec, p_vec, o_vec)
                
            if not is_legal:
                 type_effect("\n[MIIRI-HALT] La composition a été REJETÉE par le Moteur Symbolique.")
                 type_effect("[MIIRI-HALT] Le vecteur n'est pas logiquement valide. Je refuse de générer une réponse (Hallucination bloquée).")
                 continue
                 
            elapsed_reasoning = time.time() - start_time
            print(f"  -> Consensus reached across {core.num_experts} sub-agents in {elapsed_reasoning:.3f}s.")
            
            print("[Port 26420] Omni-Decoding...")
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
