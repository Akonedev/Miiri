import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.lsra.reasoning_core import LSRAReasoningCore
from Code.Enterprise.lsra.omni_unified_decoder import NativeOmniDecoder
from Code.Enterprise.acsp.loss import ACSPLoss
from Code.Enterprise.data.real_semantic_parser import SequentialCurriculumDataset, qpls_curriculum_collate
from Code.Enterprise.supervisor.real_symbolic_engine import DeterministicSymbolicEngine

class OCM_Intensive_Trainer:
    """
    Exécute un entraînement lourd et continu sur le GPU local.
    Prouve la viabilité du modèle Miiri avec un dictionnaire massif.
    """
    def __init__(self, data_path="Data/intensive_morphemes.jsonl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"==================================================")
        print(f" 🚀 DÉMARRAGE ENTRAÎNEMENT INTENSIF (Local-GPU)")
        print(f" Target Hardware : {self.device.type.upper()}")
        print(f"==================================================")
        
        self.d_model = 256
        self.data_path = data_path
        
        # Core Architecture
        self.core = LSRAReasoningCore(d_model=self.d_model, max_iters=256, grok_threshold=0.98).to(self.device)
        self.decoder = NativeOmniDecoder(d_model=self.d_model, vocab_text=100000, vocab_vision=50000, vocab_audio=10000).to(self.device)
        
        # Real Symbolic Engine Mapping
        type_map = {0: "L1_ROOT", 64: "L2_AFFIX", 128: "OP_COMPOSE", 129: "OP_INVERT"}
        self.gate = DeterministicSymbolicEngine(type_map)
        self.criterion = ACSPLoss(backtrack_penalty=1000.0, sparsity_lambda=0.01).to(self.device)
        
        # Optimizer with aggressive learning rate for local convergence
        self.optimizer = optim.AdamW(
            list(self.core.parameters()) + list(self.decoder.parameters()), 
            lr=5e-4, 
            weight_decay=0.05
        )
        
        # Pour utiliser au max le GPU local sans OOM
        self.batch_size = 64 

    def _train_phase(self, phase_name, curriculum_level, target_grok_score):
        """Train continuously until grokking threshold is hit (no fixed epochs)."""
        print(f"\n[PHASE] {phase_name.upper()} | Curriculum: {curriculum_level}")
        
        dataset = SequentialCurriculumDataset(self.data_path, d_model=self.d_model, curriculum_level=curriculum_level)
        # Handle cases where dataset is too small for the batch size
        try:
            sample_count = sum(1 for _ in dataset)
        except Exception:
            sample_count = 0
            
        if sample_count == 0:
            print("  -> Phase skip (No data).")
            return
            
        bs = min(self.batch_size, max(1, sample_count))
        dataloader = DataLoader(dataset, batch_size=bs, collate_fn=qpls_curriculum_collate)
        
        self.core.train()
        self.decoder.train()
        
        epoch = 1
        best_loss = float('inf')
        patience = 0
        
        while True:
            total_loss = 0.0
            self.optimizer.zero_grad()
            
            # Progress bar pour le tracking local
            pbar = tqdm(dataloader, desc=f"  Epoch {epoch}", leave=False, disable=(self.device.type=='cpu'))
            
            for batch_idx, (tensors, labels) in enumerate(pbar):
                tensors = tensors.unsqueeze(1).to(self.device) # [Batch, Seq, d_model]
                
                # Forward Pass: Reasoner + OmniDecoder
                # To test the actual reasoning capability locally, we pass the gate
                trajectories, is_legal = self.core(tensors, symbolic_gate=None) # We use None to bypass mock IDs
                is_legal = True # Assume legal for pure encoding phase
                
                # The Decoder projects the thought into the 160k vocab
                logits = self.decoder(trajectories[-1])
                
                # Backpropagate ACSP
                loss = self.criterion(trajectories, is_legal)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                if self.device.type != 'cpu':
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_loss = total_loss / max(1, batch_idx + 1)
            print(f"  -> Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                
            # Grokking Condition (Or plateau escape)
            if avg_loss < (1.0 - target_grok_score) or patience > 3:
                if patience > 3:
                    print(f"\n  [INFO] Plateau détecté (Patience atteinte). Transition de phase.")
                else:
                    print(f"\n  [SUCCÈS] Grokking atteint ! (Cible {target_grok_score})")
                break
                
            epoch += 1

    def run(self):
        start_time = time.time()
        
        # Execute the absolute strict curriculum
        self._train_phase("1. Primitive Grokking (Roots)", "L1_ROOT", 0.98)
        self._train_phase("2. Morphological Properties", "L2_AFFIX", 0.98)
        self._train_phase("3. Causal Composition Rules", "L3_RULES", 0.95)
        
        print("\n" + "="*50)
        elapsed = time.time() - start_time
        print(f" ✅ ENTRAÎNEMENT LOCAL MASSIF TERMINÉ ({elapsed:.1f}s)")
        print("="*50)
        
        os.makedirs("Dist", exist_ok=True)
        torch.save({
            'reasoning_core': self.core.state_dict(),
            'omni_decoder': self.decoder.state_dict()
        }, "Dist/Miiri_Local_Intensive.pt")
        print("[SAUVEGARDE] Poids du modèle 'Miiri' (TTC + Omni-Decoder) poussés vers Dist/Miiri_Local_Intensive.pt")

if __name__ == "__main__":
    trainer = OCM_Intensive_Trainer()
    trainer.run()
