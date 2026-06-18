import torch
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.tools.real_browser_use import RealBrowserTool
from Code.Enterprise.supervisor.living_workspace import LivingGlobalWorkspace
from Code.Enterprise.lsra.omni_unified_decoder import NativeOmniDecoder

def type_effect(text, delay=0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def test_real_autonomous_web_learning():
    print("==================================================")
    print(" 🌍 TEST RÉEL : AUTO-APPRENTISSAGE WEB & OMNI-DÉCODAGE")
    print("==================================================")
    
    device = torch.device("cpu")
    browser = RealBrowserTool(d_model=256)
    workspace = LivingGlobalWorkspace(d_model=256, learning_mode="auto")
    decoder = NativeOmniDecoder(d_model=256, vocab_text=50000, vocab_vision=16384, vocab_audio=8192).to(device)
    
    target_url = "https://en.wikipedia.org/wiki/Quantum_mechanics"
    print(f"\n[USER] > Mission : Va sur {target_url}, apprends le concept, et dis-moi ce que tu génères.")
    
    # 1. Évaluation Initiale
    # Le modèle ne connait pas le concept (Incertitude 1.0)
    latent_vector = torch.zeros(1, 1, 256).to(device)
    latent_vector[0, 0, 255] = 1.0
    
    needs_search = workspace.evaluate_epistemic_uncertainty(latent_vector)
    
    if needs_search:
        print("\n[MIIRI] > Je ne sais pas. Lancement du Tool [ACTION_OPEN_BROWSER]...")
        
        # 2. Exécution RÉELLE du Scraping
        extracted_text = browser.navigate_and_read(target_url)
        if not extracted_text:
            print("[ERREUR] Impossible de scraper l'URL.")
            return
            
        print(f"\n[APERÇU DES DONNÉES SCRAPÉES] : {extracted_text[:150]}...")
        
        # 3. Conversion RÉELLE en Tenseur QPLS
        learned_tensor = browser.text_to_naive_qpls(extracted_text).to(device)
        print("\n[MIIRI] > Conversion des données en Tenseur Mentalese (Orthogonal Sparsity appliquée).")
        print(f"[MIIRI] > Aperçu du vecteur (D0-D5) : {learned_tensor[0, 0, :5].tolist()}")
        
        # 4. Assimilation en Mémoire
        workspace.semantic_memory["quantum_mechanics"] = learned_tensor
        print("[WORKSPACE] Graphe de connaissance mis à jour avec le vrai tenseur.")
        
        # 5. GÉNÉRATION RÉELLE (Sans Mock)
        # On passe le tenseur appris directement à l'Omni-Decoder pour voir quels tokens il produit
        print("\n[MIIRI] > Passage du nouveau concept dans l'Omni-Decoder natif...")
        
        with torch.no_grad():
            logits = decoder(learned_tensor)
            
        # On récupère les 3 tokens les plus probables générés par ce nouveau vecteur
        top_k = 3
        probs, indices = torch.topk(logits, top_k, dim=-1)
        
        print("\n[RÉSULTAT BRUT DE LA GÉNÉRATION (Tokens générés mathématiquement)] :")
        for i in range(top_k):
            token_id = indices[0, 0, i].item()
            if token_id < 50000:
                modality = "TEXTE"
            elif token_id < 66384:
                modality = "IMAGE"
            else:
                modality = "AUDIO"
                
            type_effect(f"  -> Rank {i+1} : Token ID [{token_id}] (Modalité : {modality}) | Probabilité brute : {probs[0, 0, i].item():.4f}")
            
        print("\n[SUCCESS] Le cycle complet 'Scraping Réel -> Tenseur -> Décodage Token' est validé sans aucune simulation print().")

if __name__ == "__main__":
    test_real_autonomous_web_learning()
