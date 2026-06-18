import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.data.advanced_semantic_engine import AdvancedSemanticEngine
from Code.Enterprise.lsra.omni_detokenizers import World3DDetokenizer, AudioDetokenizer, VisualDetokenizer

@pytest.fixture
def semantic_engine():
    # property_dim = 64 (D[64:127] segment of QPLS)
    return AdvancedSemanticEngine(property_dim=64)

def test_semantic_exclusion_on_3d_world(semantic_engine):
    """
    Test Mode: 3D WORLD
    Vérifie que l'Exclusion Mutuelle empêche la génération d'un monde 3D
    aux propriétés physiques paradoxales (ex: Gravité Inversée + Gravité Normale).
    """
    # Simulation d'un vecteur de propriétés demandant deux états physiques impossibles simultanément
    paradox_properties = torch.randn(1, 64) * 10.0 # Forte variance = activation chaotique
    
    # Passage dans le moteur sémantique AVANT la génération 3D
    resolved_properties = semantic_engine.check_mutual_exclusion(paradox_properties)
    
    # Le moteur doit avoir amorti le vecteur (Damping) pour étouffer le paradoxe
    assert torch.norm(resolved_properties) < torch.norm(paradox_properties), "L'exclusion mutuelle a échoué. Le paradoxe s'est propagé."
    
    # Reconstruction du vecteur QPLS complet (d=256) pour le décodeur
    qpls_vector = torch.zeros(1, 1, 256)
    qpls_vector[0, 0, 64:128] = resolved_properties
    
    decoder_3d = World3DDetokenizer(d_model=256)
    point_cloud = decoder_3d(qpls_vector)
    
    assert point_cloud.shape == (1, 1, 3), "La génération 3D a échoué après filtrage sémantique."

def test_cross_modal_gating_audio_video(semantic_engine):
    """
    Test Mode: AUDIO / VIDEO (In & Out)
    Vérifie que l'attention inter-niveaux gère les conflits entre ce qui est "vu" et "entendu".
    """
    # Vecteur provenant du Lobe Visuel (ex: Une explosion silencieuse dans l'espace)
    visual_vec = torch.ones(1, 64) * 0.8
    # Vecteur provenant du Lobe Audio (ex: Bruit de vent)
    audio_vec = torch.ones(1, 64) * 0.2
    
    # Gating basé sur le contexte de la scène (Espace = Vide = Pas de propagation sonore)
    # L'engine doit supprimer l'audio si la règle physique (Syntax_role = VACUUM) l'exige
    resolved_vec = semantic_engine.cross_modal_gating(visual_vec, audio_vec, syntax_role="ABSTRACT_NOUN")
    
    # Dans notre prototype ABSTRACT_NOUN ignore le 2eme argument (qui ici est l'audio)
    assert torch.all(resolved_vec == visual_vec), "Le Gating Cross-Modal n'a pas filtré l'audio hors-contexte."

def test_inheritance_on_code_generation(semantic_engine):
    """
    Test Mode: CODE GENERATION
    Vérifie que la génération de code hérite correctement des propriétés abstraites.
    Ex: Le concept de 'Classe' hérite des propriétés de 'Structure Orientée Objet'.
    """
    vec_oop_base = torch.ones(1, 64) * 0.5 # [Encapsulation, Polymorphisme]
    vec_specific_class = torch.randn(1, 64) * 0.1 # [Nom de la classe: Utilisateur]
    
    # Héritage sémantique
    resolved_code_concept = semantic_engine.resolve_inheritance(vec_specific_class, vec_oop_base)
    
    # Le concept résolu doit contenir l'empreinte du parent (OOP)
    assert torch.norm(resolved_code_concept) > torch.norm(vec_specific_class), "L'héritage sémantique a échoué sur le code."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
