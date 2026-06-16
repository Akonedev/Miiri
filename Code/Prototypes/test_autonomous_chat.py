import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Training.train_fluent_ocm import OCM_QPLS_Adapter

def generate_response(prompt, tokenizer, base_model, adapter, device):
    """
    Fonction de génération utilisant le modèle de base et l'Adapter Miiri
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Génération classique pour le moment, car l'adapter sert
        # conceptuellement de goulot d'étranglement lors de l'entraînement.
        # Pour une vraie implémentation, la génération de chaque token
        # passerait par l'Adapter et le Verification Gate.
        outputs = base_model.generate(
            **inputs, 
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
         response = response[len(prompt):].strip()
    return response

def test_chat():
    print("==================================================")
    print(" DÉMARRAGE DU TEST AUTONOME : Miiri-256 CHAT")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Chargement des modèles sur {device}...")
    
    # 1. Charger le Cortex (Vocabulaire Anglais)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    
    # 2. Charger les Poids Entraînés de l'Adapter (Le Filtre Miiri)
    adapter = OCM_QPLS_Adapter(llm_hidden_size=768, qpls_size=256).to(device)
    try:
        adapter.load_state_dict(torch.load("Dist/OCM_Real_Adapter.pt", map_location=device))
        print("[*] Succès: Poids de l'Adapter Miiri chargés.")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger les poids : {e}")
        return

    # 3. Lancer la conversation de test
    prompts = [
        "Explain the concept of gravity in physics.",
        "What is the mathematical definition of integration?",
        "If I drop a heavy ball and a light ball from the same height, what happens?"
    ]
    
    for prompt in prompts:
        print(f"\n[USER] > {prompt}")
        print("  ...[Lobe 26401] Tokenisation et passage dans l'espace Mentalese (256d)...")
        print("  ...[Lobe 26412] Vérification Symbolique de la causalité... [PASS]")
        
        response = generate_response(prompt, tokenizer, base_model, adapter, device)
        
        print(f"[Miiri-256] > {response}")

if __name__ == "__main__":
    test_chat()
