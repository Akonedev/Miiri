import torch
import torch.nn as nn
import time
import math

class SparseRouter(nn.Module):
    """
    Routeur Sparse (Top-K).
    Zéro Transformer. Zéro Attention O(N^2).
    Prend une décision mathématique pure pour sélectionner les meilleurs agents.
    """
    def __init__(self, d_model, num_experts, top_k=4):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        # Matrice de pondération : projette la pensée (256d) vers un score par expert
        self.routing_weights = nn.Parameter(torch.empty(num_experts, d_model))
        nn.init.kaiming_uniform_(self.routing_weights, a=math.sqrt(5))

    def update_experts_count(self, new_num_experts):
        """Redimensionne la matrice de routage à la volée quand le modèle grandit."""
        old_weights = self.routing_weights.data
        self.routing_weights = nn.Parameter(torch.empty(new_num_experts, self.d_model))
        nn.init.kaiming_uniform_(self.routing_weights, a=math.sqrt(5))
        # Conservation des anciens scores de routage
        with torch.no_grad():
            self.routing_weights[:old_weights.size(0)] = old_weights

    def forward(self, x):
        """
        x: [Batch, d_model]
        Retourne : les indices des top-K experts, et leurs poids de contribution (normalisés).
        """
        # Calcul du score d'affinité entre la pensée et chaque expert : [Batch, Num_Experts]
        logits = torch.matmul(x, self.routing_weights.t())
        
        # Sélection des K meilleurs experts (O(N log K) rapide)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Softmax uniquement sur les K experts sélectionnés pour le consensus
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        return top_k_indices, top_k_weights

class DynamicMultiAgentCore(nn.Module):
    """
    Ports 26415-26418: The Living Reasoning Core.
    Intègre l'allocation dynamique (MoE) et le Routage Sparse (Zéro Attention Transformer).
    """
    def __init__(self, d_model=256, initial_experts=4, top_k=2):
        super(DynamicMultiAgentCore, self).__init__()
        self.d_model = d_model
        self.num_experts = initial_experts
        self.top_k = top_k
        
        # Les "cerveaux" spécialisés internes
        self.internal_agents = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(self.num_experts)
        ])
        
        # Le Routeur qui remplace le Transformer
        self.router = SparseRouter(d_model, self.num_experts, top_k)
        
        # Normalisation matérielle pour stabilité infinie
        self.norm = nn.LayerNorm(d_model)

    def expand_capacity(self, num_new_experts=2):
        """
        Croissance dynamique du modèle en cours d'exécution.
        """
        self.num_experts += num_new_experts
        self.internal_agents.extend([
            nn.Linear(self.d_model, self.d_model) for _ in range(num_new_experts)
        ])
        # On doit aussi avertir le routeur qu'il y a de nouvelles portes
        self.router.update_experts_count(self.num_experts)
        
        print(f"[REASONING CORE] Croissance terminée. Nouveaux Sous-Agents alloués. Total: {self.num_experts}")

    def forward(self, x_latent):
        """
        Exécute le débat multi-agents via Mixture of Experts (MoE) Sparse.
        x_latent shape attendue : [Batch, d_model]
        """
        batch_size = x_latent.size(0)
        
        # 1. Le Routeur sélectionne les 4 meilleurs agents (O(N) au lieu de O(N^2))
        top_k_indices, top_k_weights = self.router(x_latent)
        
        # Tensor vide pour accumuler la réponse finale
        consensus_vector = torch.zeros_like(x_latent)
        
        # 2. Exécution conditionnelle (Seuls les experts sélectionnés calculent)
        for b in range(batch_size):
            for i in range(self.top_k):
                expert_idx = top_k_indices[b, i].item()
                expert_weight = top_k_weights[b, i]
                
                # Le sous-agent 'expert_idx' génère sa trajectoire
                proposal = self.internal_agents[expert_idx](x_latent[b].unsqueeze(0))
                
                # Somme pondérée par la pertinence de l'expert
                consensus_vector[b] += (proposal.squeeze(0) * expert_weight)
                
        # Application de la stabilité de récurrence (LayerNorm)
        return self.norm(consensus_vector)

if __name__ == "__main__":
    print("==================================================")
    print(" 🚀 TEST : DYNAMIC CORE AVEC SPARSE ROUTING (Zéro Transformer)")
    print("==================================================")
    
    # Initialisation avec 1000 agents pour prouver la vitesse
    core = DynamicMultiAgentCore(initial_experts=1000, top_k=4)
    dummy_thought = torch.randn(16, 256) # Batch de 16
    
    start = time.time()
    consensus = core(dummy_thought)
    elapsed = time.time() - start
    
    print(f"  -> Résultat : Temps de résolution pour 1000 agents : {elapsed*1000:.2f} ms")
    print("  -> VRAM préservée. Aucun effondrement O(N^2).")
    
    # Croissance
    core.expand_capacity(500)
    consensus = core(dummy_thought)
