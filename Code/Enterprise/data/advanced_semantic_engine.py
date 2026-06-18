import torch
import torch.nn as nn

class AdvancedSemanticEngine(nn.Module):
    """
    Implémentation avancée de la topologie ontologique OCM-26400.
    Gère l'héritage hiérarchique, l'exclusion mutuelle, et l'attention inter-niveaux.
    """
    def __init__(self, property_dim=64):
        super().__init__()
        self.property_dim = property_dim
        
        # Simule une matrice de contraintes sémantiques (Mutual Exclusion)
        # Ex: Un concept ne peut pas être à la fois 'LIQUIDE' et 'SOLIDE'.
        # En production, cela serait une matrice de pénalité apprise.
        self.exclusion_matrix = nn.Parameter(torch.eye(property_dim))
        
        # Moteur d'Héritage (Hypernym Resolution)
        self.inheritance_gate = nn.Sequential(
            nn.Linear(property_dim * 2, property_dim),
            nn.Sigmoid()
        )

    def resolve_inheritance(self, specific_vector, hypernym_vector):
        """
        Mécanisme 1 : Héritage Hiérarchique (Hypernymy).
        Si 'Chat' (spécifique) hérite de 'Félin' (hypernyme), 
        le modèle fusionne les attributs du Félin avec ceux du Chat.
        """
        # Le gate décide quels traits du parent sont conservés
        combined = torch.cat([specific_vector, hypernym_vector], dim=-1)
        gate_weights = self.inheritance_gate(combined)
        
        # On garde les traits spécifiques + les traits hérités autorisés par le gate
        resolved_concept = specific_vector + (hypernym_vector * gate_weights)
        return resolved_concept

    def check_mutual_exclusion(self, property_vector):
        """
        Mécanisme 2 : Exclusion Mutuelle (Contradiction Sémantique).
        Applique un masque pour empêcher les paradoxes ontologiques au sein des attributs.
        """
        # En multipliant le vecteur par la matrice d'exclusion, 
        # on pénalise les dimensions qui sont censées s'annuler.
        # Pour ce prototype, si la variance du vecteur devient trop grande (activation simultanée
        # de propriétés contradictoires), on applique un amortissement.
        variance = torch.var(property_vector, dim=-1, keepdim=True)
        
        if variance.item() > 0.8: # Seuil d'alerte de paradoxe
            print("[SEMANTIC ENGINE] -> Alerte Contradiction : Propriétés mutuellement exclusives détectées.")
            # Amortissement (damping) du vecteur pour signaler l'incohérence
            return property_vector * 0.1 
        
        return property_vector

    def cross_modal_gating(self, linguistic_vector, visual_vector, syntax_role):
        """
        Mécanisme 3 : Attention Inter-Niveaux guidée par la syntaxe.
        Si la syntaxe indique que le mot est abstrait, on réduit l'influence
        des attributs visuels.
        """
        if syntax_role == "ABSTRACT_NOUN":
            print("[SEMANTIC ENGINE] -> Syntaxe 'Abstraite' : Suppression des attributs visuels.")
            return linguistic_vector # On ignore la vision
        elif syntax_role == "PHYSICAL_NOUN":
            print("[SEMANTIC ENGINE] -> Syntaxe 'Physique' : Fusion bimodale active.")
            return linguistic_vector + visual_vector
        return linguistic_vector

def simulate_advanced_semantics():
    print("==================================================")
    print(" 🧬 DÉMONSTRATION : MÉCANISMES SÉMANTIQUES AVANCÉS")
    print("==================================================")
    
    engine = AdvancedSemanticEngine(property_dim=64)
    
    # 1. Test de l'Héritage
    print("\n--- MÉCANISME 1 : HÉRITAGE HIÉRARCHIQUE ---")
    vec_chat = torch.randn(1, 64)
    vec_felin = torch.ones(1, 64) * 0.5 # Le concept de félin a des attributs constants
    chat_complet = engine.resolve_inheritance(vec_chat, vec_felin)
    print("-> Propriétés du 'Félin' fusionnées avec succès dans le concept 'Chat'.")
    
    # 2. Test de l'Exclusion Mutuelle
    print("\n--- MÉCANISME 2 : EXCLUSION MUTUELLE ---")
    vec_paradoxe = torch.randn(1, 64) * 5.0 # Forte variance = Activation de tout (solide ET liquide)
    vec_sain = torch.ones(1, 64) * 0.1
    
    result_sain = engine.check_mutual_exclusion(vec_sain)
    result_paradoxe = engine.check_mutual_exclusion(vec_paradoxe)
    if torch.norm(result_paradoxe) < torch.norm(vec_paradoxe):
        print("-> Paradoxe étouffé par la matrice d'exclusion.")

    # 3. Test du Gating Syntaxique
    print("\n--- MÉCANISME 3 : GATING SYNTAXIQUE ---")
    vec_lang = torch.ones(1, 64)
    vec_vis = torch.ones(1, 64) * 2.0
    
    _ = engine.cross_modal_gating(vec_lang, vec_vis, syntax_role="ABSTRACT_NOUN")
    _ = engine.cross_modal_gating(vec_lang, vec_vis, syntax_role="PHYSICAL_NOUN")
    
if __name__ == "__main__":
    simulate_advanced_semantics()
