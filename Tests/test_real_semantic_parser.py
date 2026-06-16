import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.data.real_semantic_parser import SequentialCurriculumDataset, qpls_curriculum_collate, AntiShortcutMasker

def test_sequential_curriculum_dataloader():
    data_path = "Data/real_morphemes.jsonl"
    
    if not os.path.exists(data_path):
        print("Dataset non trouvé. Génération en cours...")
        from Code.Enterprise.data.generate_real_data import generate_real_linguistic_dataset
        generate_real_linguistic_dataset(data_path)
        
    print("\n[TEST] 1. Test du filtre 'Sequential Grokking' (L1_ROOT)...")
    dataset_l1 = SequentialCurriculumDataset(data_path, d_model=256, curriculum_level="L1_ROOT")
    loader_l1 = DataLoader(dataset_l1, batch_size=2, collate_fn=qpls_curriculum_collate)
    
    for batch, labels in loader_l1:
        assert batch.shape[-1] == 256
        # Vérification que ce sont bien uniquement des racines
        for label in labels:
            assert label in ["chron", "log", "therm", "meter"]
        print(f"  -> Validé: Batch L1_ROOT chargé proprement. Contient: {labels}")

    print("\n[TEST] 2. Test du filtre 'Sequential Grokking' (L2_AFFIX)...")
    dataset_l2 = SequentialCurriculumDataset(data_path, d_model=256, curriculum_level="L2_AFFIX")
    loader_l2 = DataLoader(dataset_l2, batch_size=2, collate_fn=qpls_curriculum_collate)
    
    for batch, labels in loader_l2:
        assert batch.shape[-1] == 256
        for label in labels:
            assert label in ["ology", "anti", "ic"]
        print(f"  -> Validé: Batch L2_AFFIX chargé proprement. Contient: {labels}")

    print("\n[TEST] 3. Test du 'Anti-Shortcut Masking Law'...")
    # On simule un vecteur "réponse"
    ans_vector = torch.ones(1, 256)
    masked_vector = AntiShortcutMasker.apply_algebraic_mask(ans_vector, mask_probability=1.0)
    assert torch.all(masked_vector == 0), "L'Anti-Shortcut Masker n'a pas masqué la réponse."
    print("  -> Validé: Cible masquée avec succès pour empêcher le raccourci algébrique c = ans - m1.")

if __name__ == "__main__":
    print("==================================================")
    print(" AUDIT: PRODUCTION DATA PIPELINE & CURRICULUM")
    print("==================================================")
    test_sequential_curriculum_dataloader()
    print("\n[SUCCESS] Tests du Data Pipeline terminés avec succès.")
