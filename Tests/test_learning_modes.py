import torch
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.supervisor.living_workspace import LivingGlobalWorkspace

def run_learning_mode_test(mode):
    print(f"\n==================================================")
    print(f" 🧪 TEST D'APPRENTISSAGE : MODE {mode.upper()}")
    print(f"==================================================")
    
    workspace = LivingGlobalWorkspace(learning_mode=mode)
    device = torch.device("cpu")
    
    question = "Quelle est la capitale de la France ?"
    print(f"[USER] > {question}")
    
    # 1. Le modèle génère un vecteur. Il ne connait pas la réponse (Simulé)
    latent_vector = torch.zeros(1, 1, 256).to(device)
    latent_vector[..., 255] = 1.0 # Incertitude maximale
    
    # 2. Évaluation
    needs_search = workspace.evaluate_epistemic_uncertainty(latent_vector)
    
    if needs_search:
        print("[MIIRI] > Je ne connais pas la réponse. Je lance une recherche web (TOOL_USE)...")
        time.sleep(1)
        
        # 3. Scraping simulé
        search_result = "Paris est la capitale de la France."
        urls = ["https://fr.wikipedia.org/wiki/France"]
        
        print(f"[MIIRI] > Résultats trouvés : '{search_result}' (Source: {urls[0]})")
        
        # 4. Le Workspace gère l'apprentissage selon le mode
        learned = workspace.handle_unknown_concept(search_result, urls)
        
        if learned:
            print(f"[MIIRI] > {search_result}")
        else:
            print(f"[MIIRI] > J'ai trouvé des informations, mais l'apprentissage a été refusé. Je ne peux pas répondre avec certitude.")

    print("\n--- DEUXIÈME TENTATIVE (Le modèle doit se souvenir s'il a appris) ---")
    print(f"[USER] > {question}")
    
    # 5. Vérification de la mémorisation
    memorized_fact = workspace.query_semantic_memory()
    
    if memorized_fact:
        # L'incertitude est maintenant de 0.0
        latent_vector[..., 255] = 0.0
        print(f"[MIIRI] > (Rappel Immédiat Zéro-Shot) : {memorized_fact}")
    else:
        print(f"[MIIRI] > Je ne connais toujours pas la réponse.")

if __name__ == "__main__":
    # Test Mode Auto
    run_learning_mode_test("auto")
    
    # Test Mode Supervisé (Require User Input)
    run_learning_mode_test("supervised")
