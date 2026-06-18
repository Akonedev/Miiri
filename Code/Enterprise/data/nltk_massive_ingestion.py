import os
import sys
import time
import json
import torch
import nltk
from nltk.corpus import wordnet as wn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.data.dynamic_vocabulary_engine import DynamicVocabularyEngine
from Code.Enterprise.data.omni_shared_field import OmniSharedField

def setup_nltk():
    print("[*] Téléchargement des bases de données linguistiques réelles (WordNet, OMW)...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True) # Open Multilingual WordNet (pour le Français)
    print("[*] Bases de données téléchargées.")

def extract_linguistic_features(word, lang='eng'):
    """
    Extrait TOUTES les caractéristiques exigées (Grammaire, Sens, Synonymes, etc.)
    en une seule passe à partir de WordNet.
    """
    synsets = wn.synsets(word, lang=lang)
    if not synsets:
        return None

    # On prend le sens principal (le premier synset)
    primary_synset = synsets[0]
    
    features = {
        "word": word,
        "language": lang,
        "definition": primary_synset.definition(),
        "pos": primary_synset.pos(), # Part of Speech (n=noun, v=verb, a=adj, r=adv)
        "lemmas": [lemma.name() for lemma in primary_synset.lemmas(lang=lang)], # Radicaux/Lexèmes
        "synonyms": [],
        "antonyms": [],
        "hypernyms": [h.name().split('.')[0] for h in primary_synset.hypernyms()], # Catégorie supérieure
    }

    # Extraction des synonymes et antonymes
    for lemma in primary_synset.lemmas(lang=lang):
        features["synonyms"].append(lemma.name())
        if lemma.antonyms():
            features["antonyms"].append(lemma.antonyms()[0].name())
            
    # Déduplication
    features["synonyms"] = list(set(features["synonyms"]))
    features["antonyms"] = list(set(features["antonyms"]))

    return features

def build_massive_linguistic_dataset(output_file="Data/massive_lexicon.jsonl", max_words=None):
    """
    Parcourt l'intégralité du dictionnaire WordNet et extrait les features
    pour créer le dataset d'ingestion.
    """
    setup_nltk()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    words = list(wn.words())
    total_words = len(words)
    limit = max_words if max_words else total_words
    
    print(f"[*] Début de l'extraction linguistique sur {limit} mots...")
    
    extracted_count = 0
    start_time = time.time()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, word in enumerate(words):
            if i >= limit:
                break
                
            features = extract_linguistic_features(word)
            if features:
                f.write(json.dumps(features) + "\n")
                extracted_count += 1
                
            if (i + 1) % 10000 == 0:
                print(f"  -> Traité: {i+1}/{total_words} mots...")
                
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Extraction terminée : {extracted_count} mots complets enregistrés en {elapsed:.2f}s.")
    print(f"[SUCCESS] Fichier généré : {output_file}")
    return output_file

def simulate_ingestion_pipeline(dataset_file, limit=10):
    """
    Démontre comment l'architecture Miiri ingère ces données réelles
    dans le Dynamic Vocabulary Engine et l'OmniSharedField.
    """
    print("\n==================================================")
    print(" 🧠 PIPELINE D'INGESTION LINGUISTIQUE (MÉTADONNÉES)")
    print("==================================================")
    
    vocab_engine = DynamicVocabularyEngine(initial_vocab_size=1000)
    
    print(f"[*] Lecture des {limit} premiers mots de {dataset_file}...")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            
            data = json.loads(line)
            word = data["word"]
            pos = data["pos"]
            synonyms = data["synonyms"]
            definition = data["definition"]
            
            # 1. Le mot brut entre dans le Dynamic Vocab Engine
            # Cela garantit la croissance infinie si le mot est nouveau
            vector = vocab_engine.ingest_and_encode([word])
            
            print(f"\n-> MOT INGÉRÉ : '{word}' (Grammaire: {pos})")
            print(f"   Définition : {definition}")
            print(f"   Synonymes  : {synonyms}")
            print(f"   Tenseur QPLS généré : Shape {vector.shape}")
            
            # Dans l'entraînement complet, l'OmniSharedField lierait ces ID de synonymes
            # et de grammaire au vecteur racine.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true", help="Only extract the dataset")
    parser.add_argument("--limit", type=int, default=1000, help="Number of words to extract/process for this run")
    args = parser.parse_args()

    dataset_path = build_massive_linguistic_dataset(max_words=args.limit)
    if not args.extract_only:
        simulate_ingestion_pipeline(dataset_path, limit=min(5, args.limit))
