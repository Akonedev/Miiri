import torch
from torch.utils.data import IterableDataset
import json
import os

class SequentialCurriculumDataset(IterableDataset):
    """
    DataLoader Production Grade respectant la Loi du "Sequential Grokking".
    Ne permet le chargement que des niveaux de curriculum stricts pour éviter l'interférence de gradient.
    """
    def __init__(self, file_path, d_model=256, curriculum_level="L1_ROOT"):
        super().__init__()
        self.file_path = file_path
        self.d_model = d_model
        self.curriculum_level = curriculum_level
        assert self.d_model in [256, 512], "Architecture strictly expects d_model 256 or 512"

    def _process_line(self, line):
        data = json.loads(line)
        
        # Filtre Strict du Curriculum (Empêche l'interférence L1 / L2)
        if data['category'] != self.curriculum_level:
            return None
            
        primitive_type = data['type']
        raw_val = torch.tensor(data['semantic_hash'], dtype=torch.float32)
        
        # Pad dynamic hash to exact partition size (64 dimensions)
        segment_dim = self.d_model // 4
        padded_val = torch.zeros(segment_dim)
        padded_val[:len(raw_val)] = raw_val
        
        # Construction du tenseur QPLS
        qpls_vector = torch.zeros(self.d_model)
        
        # Orthogonal Sparsity : Injection stricte dans la bonne partition, le reste reste 0.0
        if primitive_type == "ENTITY":
            qpls_vector[0:segment_dim] = padded_val
        elif primitive_type == "PROPERTY":
            qpls_vector[segment_dim:2*segment_dim] = padded_val
        elif primitive_type == "OPERATOR":
            qpls_vector[2*segment_dim:3*segment_dim] = padded_val
            
        # Metadata (Source & Confiance initiale)
        qpls_vector[3*segment_dim] = 1.0 # D[192] = Modalité Textuelle
            
        return qpls_vector, data['morpheme']

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tensor_data = self._process_line(line)
                if tensor_data is not None:
                    yield tensor_data

class AntiShortcutMasker:
    """
    Implémente la Loi du "Masquage Multivarié".
    (c = ans - m1) exploit prevention.
    """
    @staticmethod
    def apply_algebraic_mask(target_vector, mask_probability=1.0):
        """
        Masque purement la cible algébrique pendant l'entraînement de la phase de composition
        pour forcer le modèle à calculer via les primitives, et non par rétro-ingénierie du résultat.
        """
        if torch.rand(1).item() <= mask_probability:
            # Mask the entirety of the answer vector to force true composition
            return torch.zeros_like(target_vector)
        return target_vector

def qpls_curriculum_collate(batch):
    """
    Collate fn for DataLoader that verifies Orthogonal Sparsity before GPU transfer.
    """
    tensors, labels = zip(*batch)
    stacked = torch.stack(tensors)
    d_model = stacked.shape[-1]
    seg = d_model // 4
    
    # Sécurité Production : Crash immédiat si le tenseur est "brouillé"
    for vec in stacked:
        e, p, o, m = vec[0:seg], vec[seg:2*seg], vec[2*seg:3*seg], vec[3*seg:4*seg]
        active_blocks = (e.any(), p.any(), o.any())
        assert sum(active_blocks) == 1, "FATAL: Orthogonal Sparsity violated in data pipeline!"
        
    return stacked, labels
