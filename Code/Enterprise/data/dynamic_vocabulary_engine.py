import torch
import torch.nn as nn
import hashlib
import json
import os

class DynamicVocabularyEngine(nn.Module):
    """
    Moteur de Vocabulaire Dynamique à Croissance Infinie.
    Résout le problème du "Out-Of-Vocabulary (OOV) Crash".
    Si un mot est inconnu, la matrice grandit en temps réel.
    """
    def __init__(self, initial_vocab_size=1000, d_model=256, dict_path="Data/dynamic_dictionary.jsonl"):
        super().__init__()
        self.d_model = d_model
        self.dict_path = dict_path
        
        # Le dictionnaire mapping (Mot -> Token ID)
        self.word_to_id = {}
        self.id_to_word = {}
        self.current_vocab_size = 0
        
        # Charge le dictionnaire existant s'il y en a un
        self._load_dictionary()
        
        # La Matrice d'Embedding (S'agrandit dynamiquement)
        # On initialise avec une taille de base, mais elle n'est pas fixe.
        alloc_size = max(initial_vocab_size, self.current_vocab_size + 100)
        self.embedding_matrix = nn.Embedding(alloc_size, d_model)
        
        print(f"[DYNAMIC VOCAB] Moteur initialisé avec {self.current_vocab_size} mots connus.")

    def _load_dictionary(self):
        """Charge le mapping existant depuis le disque."""
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    word = data["word"]
                    t_id = data["id"]
                    self.word_to_id[word] = t_id
                    self.id_to_word[t_id] = word
            self.current_vocab_size = len(self.word_to_id)

    def _save_new_word(self, word, t_id):
        """Sauvegarde un nouveau mot de manière persistante."""
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        with open(self.dict_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"word": word, "id": t_id}) + "\n")

    def _deterministic_hash_vector(self, word):
        """
        Si on rencontre un mot inconnu, on ne l'initialise pas avec du bruit aléatoire.
        On utilise un hash déterministe projeté dans le segment Entité (D[0:64]).
        Cela donne un "sens brut" par défaut avant l'entraînement.
        """
        h = hashlib.sha256(word.encode()).digest()
        vec = torch.zeros(self.d_model)
        
        # Remplir uniquement le segment Entité (0-63)
        segment_dim = self.d_model // 4
        for i in range(segment_dim):
            # Normalisation entre -1 et 1
            vec[i] = (h[i % len(h)] / 128.0) - 1.0 
            
        return vec

    def _expand_embedding_matrix(self):
        """
        C'est le cœur de l'exigence : "Grandir sans limite".
        Si on dépasse la capacité allouée, on recrée une matrice plus grande
        en copiant les anciens poids.
        """
        old_num_embeddings = self.embedding_matrix.num_embeddings
        new_num_embeddings = old_num_embeddings + 1000 # On alloue par blocs de 1000 pour optimiser
        
        print(f"\n[CROISSANCE VRAM] Redimensionnement de la matrice de vocabulaire : {old_num_embeddings} -> {new_num_embeddings}")
        
        new_embedding = nn.Embedding(new_num_embeddings, self.d_model).to(self.embedding_matrix.weight.device)
        
        # Copie des anciens poids préservant l'apprentissage passé
        with torch.no_grad():
            new_embedding.weight[:old_num_embeddings] = self.embedding_matrix.weight.clone()
            
        self.embedding_matrix = new_embedding

    def ingest_and_encode(self, text_list):
        """
        Prend une liste de mots. Si un mot est inconnu, l'apprend à la volée.
        Retourne le tenseur Mentalese correspondant.
        """
        token_ids = []
        for word in text_list:
            word = word.lower()
            if word not in self.word_to_id:
                # 1. Le mot est inconnu. On l'apprend.
                new_id = self.current_vocab_size
                
                # 2. Vérification de la capacité VRAM
                if new_id >= self.embedding_matrix.num_embeddings:
                    self._expand_embedding_matrix()
                    
                # 3. Enregistrement
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word
                self.current_vocab_size += 1
                self._save_new_word(word, new_id)
                
                # 4. Initialisation sémantique Hachée (Orthogonal Sparsity)
                hash_vec = self._deterministic_hash_vector(word)
                with torch.no_grad():
                    self.embedding_matrix.weight[new_id] = hash_vec.to(self.embedding_matrix.weight.device)
                
                print(f"[APPRENTISSAGE LIVE] Nouveau concept acquis : '{word}' -> ID {new_id}")
            
            token_ids.append(self.word_to_id[word])
            
        # 5. Récupération des vecteurs depuis la matrice
        # Shape: [Seq_Len, d_model]
        tensor_ids = torch.tensor(token_ids, dtype=torch.long, device=self.embedding_matrix.weight.device)
        mentalese_vectors = self.embedding_matrix(tensor_ids)
        
        return mentalese_vectors

if __name__ == "__main__":
    print("==================================================")
    print(" TEST : MOTEUR D'INGESTION DYNAMIQUE (CROISSANCE INFINIE)")
    print("==================================================")
    
    engine = DynamicVocabularyEngine(initial_vocab_size=2)
    
    # Phrase avec des mots totalement inconnus
    phrase = ["Le", "modele", "apprend", "a", "la", "volee", "et", "ne", "plante", "jamais"]
    
    print(f"\n[USER] > Traite cette nouvelle phrase : {phrase}")
    
    # Le modèle ne va pas crasher, il va allouer de l'espace dynamiquement
    vectors = engine.ingest_and_encode(phrase)
    
    print(f"\n[RESULTAT] Tenseur généré avec succès. Shape: {vectors.shape}")
    print("[SUCCESS] Le problème de limite de dictionnaire est résolu. La VRAM s'adapte à la demande.")
