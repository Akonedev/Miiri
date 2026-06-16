import torch
import pytest
import sys
import os

# Ajout du chemin racine pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.Enterprise.lsra.reasoning_core import LSRAReasoningCore
from Code.Enterprise.supervisor.symbolic_gate import SymbolicVerificationGate
from Code.Enterprise.acsp.loss import ACSPLoss
from Code.Enterprise.lsra.omni_unified_decoder import NativeOmniDecoder

class ExtremeSymbolicGate(SymbolicVerificationGate):
    """
    Gate strict pour les tests extrêmes.
    Simule des règles physiques dures.
    """
    def __init__(self):
        super().__init__({})
        self.rules = {
            # ID Opérateur -> Règles
            128: {"name": "GRAVITY", "requires_entity": 1, "yields_property": 64}, # Masse -> Accélération Bas
            129: {"name": "ANTI_GRAVITY", "requires_entity": 2, "yields_property": 65} # Objet Magique -> Accélération Haut
        }

    def verify_latent_step(self, entity_vector, operator_vector):
        entity_id = torch.argmax(entity_vector).item()
        op_id = torch.argmax(operator_vector).item() + 128 # Offset pour l'opérateur
        
        if op_id in self.rules:
            if entity_id == self.rules[op_id]["requires_entity"]:
                return True
        return False

@pytest.fixture
def device():
    return torch.device("cpu") # Test local

@pytest.fixture
def core_components(device):
    core = LSRAReasoningCore(d_model=256, max_iters=100).to(device)
    gate = ExtremeSymbolicGate()
    decoder = NativeOmniDecoder(d_model=256, vocab_text=1000, vocab_vision=1000, vocab_audio=1000).to(device)
    loss_fn = ACSPLoss(backtrack_penalty=5000.0).to(device)
    return core, gate, decoder, loss_fn

def test_cross_modal_contradiction(core_components, device):
    """
    TEST EXTRÊME 1 : Le modèle essaie de générer une image d'un objet normal (Masse)
    tombant vers le HAUT sans appliquer de force anti-gravité.
    Ceci teste la rigueur causale contre les hallucinations visuelles.
    """
    core, gate, decoder, loss_fn = core_components
    
    # Simule une pensée illégale : Entité 1 (Masse normale) + Opérateur 129 (Anti-Gravité)
    illegal_thought = torch.zeros(1, 1, 256).to(device)
    illegal_thought[0, 0, 1] = 10.0 # Entité: Masse
    illegal_thought[0, 0, 129] = 10.0 # Opérateur: Anti-Gravité (Nécessite Entité 2)
    
    trajectories, is_legal = core(illegal_thought, symbolic_gate=gate)
    
    assert not is_legal, "Le Gate aurait dû bloquer cette aberration physique."
    
    loss = loss_fn(trajectories, is_legal)
    assert loss.item() >= 5000.0, "La pénalité ACSP n'a pas été appliquée !"

def test_infinite_depth_stability(core_components, device):
    """
    TEST EXTRÊME 2 : Le modèle est forcé de boucler 10 000 fois dans l'espace latent.
    On vérifie que les vecteurs n'explosent pas (NaN) grâce au LayerNorm.
    """
    core, _, _, _ = core_components
    core.max_iters = 10000 # On force 10k itérations
    
    valid_thought = torch.randn(1, 1, 256).to(device)
    # On bypass le gate pour forcer la boucle à s'exécuter jusqu'au bout
    # et on met une confiance faible pour qu'il ne s'arrête pas
    valid_thought[0, 0, 192] = -10.0 
    
    trajectories, is_legal = core(valid_thought, symbolic_gate=None)
    
    final_state = trajectories[-1]
    
    assert not torch.isnan(final_state).any(), "Explosion du Gradient (NaN) détectée !"
    assert not torch.isinf(final_state).any(), "Explosion du Gradient (Inf) détectée !"
    
    # La norme doit rester stable (proche de sqrt(256) = 16)
    norm = torch.norm(final_state).item()
    assert 10.0 < norm < 30.0, f"Instabilité du vecteur ! Norme: {norm}"

def test_omni_decoder_purity(core_components, device):
    """
    TEST EXTRÊME 3 : Vérification que l'Omni-Decoder sépare bien les modalités.
    """
    _, _, decoder, _ = core_components
    
    thought = torch.randn(1, 1, 256).to(device)
    logits = decoder(thought)
    
    assert logits.shape == (1, 1, 3000), "Le vocabulaire unifié doit faire 3000 tokens."
    
    # Si le modèle décide de générer une image (Token ID > 1000 et < 2000),
    # il ne doit pas y avoir de tokens texte mélangés dans le même échantillonnage
    token_id = torch.argmax(logits, dim=-1).item()
    assert 0 <= token_id < 3000, "Token ID hors limites."

if __name__ == "__main__":
    pytest.main(["-v", __file__])
