import json
import os
import torch
import math
import hashlib

def create_intensive_dataset(filepath="Data/intensive_morphemes.jsonl", size=1000):
    """
    Génère un dataset sémantique massif (jusqu'à `size` entrées)
    pour saturer le GPU local en émulant un dictionnaire complet.
    L'espace latent d=256 est strictement partitionné.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"[*] Génération de {size} primitives linguistiques réelles en cours...")
    
    # Mots et affixes de base pour générer les combinaisons
    roots = ["chrono", "log", "therm", "meter", "bio", "geo", "path", "psych", "neuro", "astro"]
    affixes = ["ology", "anti", "ic", "ism", "ist", "ation", "able", "ness", "ity", "ment"]
    
    data = []
    
    # Fonction de hachage déterministe pour générer des tenseurs QPLS uniques et stables
    def generate_hash_vector(word, segment_size=64):
        h = hashlib.sha256(word.encode()).digest()
        # Normalisation entre -1 et 1
        vec = [(b / 128.0) - 1.0 for b in h[:segment_dim]]
        # Padding
        if len(vec) < segment_size:
            vec.extend([0.0] * (segment_size - len(vec)))
        return vec

    segment_dim = 64
    
    # 1. Génération des Entities (L1_ROOT)
    for root in roots:
        data.append({
            "id": f"root_{root}", "type": "ENTITY", "category": "L1_ROOT", 
            "morpheme": root, "meaning": f"Concept of {root}", 
            "semantic_hash": generate_hash_vector(root, segment_dim)
        })
        
    # 2. Génération des Properties (L2_AFFIX)
    for affix in affixes:
        data.append({
            "id": f"affix_{affix}", "type": "PROPERTY", "category": "L2_AFFIX", 
            "morpheme": affix, "meaning": f"Modifies to {affix}", 
            "semantic_hash": generate_hash_vector(affix, segment_dim)
        })

    # 3. Expansion artificielle pour stresser le GPU (Multiplication des racines)
    # On ajoute des concepts "dérivés" pour augmenter la taille du dictionnaire
    for i in range(size - len(data) - 2):
        concept = f"concept_{i}"
        data.append({
            "id": concept, "type": "ENTITY", "category": "L1_ROOT", 
            "morpheme": concept, "meaning": f"Generated Entity {i}", 
            "semantic_hash": generate_hash_vector(concept, segment_dim)
        })
        
    # 4. Opérateurs de Composition (L3_RULES)
    data.append({"id": "op_compose", "type": "OPERATOR", "category": "L3_RULES", "morpheme": "COMPOSE", "meaning": "Merge", "semantic_hash": [1.0] * segment_dim})
    data.append({"id": "op_invert", "type": "OPERATOR", "category": "L3_RULES", "morpheme": "INVERT", "meaning": "Opposite", "semantic_hash": [-1.0] * segment_dim})

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"[SUCCESS] Dataset massif créé : {filepath} ({len(data)} entrées)")

if __name__ == "__main__":
    create_intensive_dataset(size=1000)
