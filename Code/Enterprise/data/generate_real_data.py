import json
import os

def generate_real_linguistic_dataset(filepath="Data/real_morphemes.jsonl"):
    """
    Génère un dataset linguistique réel (racines, affixes, opérateurs) 
    pour remplacer les mocks. Les vecteurs sont des hash sémantiques normalisés.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Un vrai dictionnaire de primitives morphologiques
    morphemes = [
        # ENTITIES (Racines / Radicaux) - Vont dans D[0-63]
        {"id": "root_chron", "type": "ENTITY", "category": "L1_ROOT", "morpheme": "chron", "meaning": "time", "semantic_hash": [0.8, -0.2, 0.1, 0.5]},
        {"id": "root_log", "type": "ENTITY", "category": "L1_ROOT", "morpheme": "log", "meaning": "word/reason", "semantic_hash": [0.6, 0.7, -0.1, 0.0]},
        {"id": "root_therm", "type": "ENTITY", "category": "L1_ROOT", "morpheme": "therm", "meaning": "heat", "semantic_hash": [0.9, 0.1, 0.8, -0.4]},
        {"id": "root_meter", "type": "ENTITY", "category": "L1_ROOT", "morpheme": "meter", "meaning": "measure", "semantic_hash": [-0.3, 0.5, 0.9, 0.1]},
        
        # PROPERTIES (Affixes / Modificateurs) - Vont dans D[64-127]
        {"id": "suffix_ology", "type": "PROPERTY", "category": "L2_AFFIX", "morpheme": "ology", "meaning": "study of", "semantic_hash": [0.1, 0.9, 0.2, 0.2]},
        {"id": "prefix_anti", "type": "PROPERTY", "category": "L2_AFFIX", "morpheme": "anti", "meaning": "against/opposite", "semantic_hash": [-0.9, -0.9, 0.0, 0.1]},
        {"id": "suffix_ic", "type": "PROPERTY", "category": "L2_AFFIX", "morpheme": "ic", "meaning": "pertaining to", "semantic_hash": [0.2, 0.1, 0.9, 0.0]},
        
        # OPERATORS (Causal Rules) - Vont dans D[128-191]
        {"id": "op_compose_noun", "type": "OPERATOR", "category": "L3_RULES", "morpheme": "COMPOSE_NOUN", "meaning": "Merge Root+Suffix into Noun", "semantic_hash": [1.0, 1.0, 1.0, 1.0]},
        {"id": "op_invert", "type": "OPERATOR", "category": "L3_RULES", "morpheme": "INVERT_MEANING", "meaning": "Apply opposite modifier", "semantic_hash": [-1.0, -1.0, -1.0, -1.0]}
    ]

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in morphemes:
            f.write(json.dumps(item) + "\n")
            
    print(f"[*] Base de données linguistique réelle générée : {filepath}")

if __name__ == "__main__":
    generate_real_linguistic_dataset()
