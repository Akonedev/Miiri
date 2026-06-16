# Chapitre 2 : Fondations Mathématiques et Algorithmes

Ce chapitre explique de manière détaillée, pour les chercheurs et "Tuners", la formulation de la Loss et du Vecteur d'Espace Latent.

## 2.1 QPLS Vector : Quad-Partitioned Latent Semantic Space
Pour permettre la vérification symbolique, l'espace latent $v \in \mathbb{R}^{256}$ est divisé de manière stricte (hard-partitioning) :
$$ v = [v_{ent} \parallel v_{prop} \parallel v_{op} \parallel v_{meta}] $$
Chaque segment est de dimension 64. 
- **L'utilité du typage dur :** Cela garantit que lors d'un produit scalaire ou d'une matrice d'attention, le modèle n'entremêle pas les "actions" avec les "objets". Une erreur de typage sémantique peut être détectée par une simple norme $L_2$ sur le segment fautif.

## 2.2 ACSP : Amodal Consistency & Step Penalty
L'ACSP est l'algorithme d'apprentissage par renforcement inversé. 
La perte totale $\mathcal{L}_{CausalRigor}$ :
$$ \mathcal{L}_{CausalRigor} = \alpha \mathcal{L}_{align} + \beta \mathcal{L}_{step} + \gamma \mathcal{L}_{sparse} + \delta \mathcal{L}_{consist} $$

### Détail des composantes et de leurs impacts :
1. **$\mathcal{L}_{step}$ (Pénalité de Backtrack)**
   Le modèle propose une transition $v^{(t)} \rightarrow v^{(t+1)}$. Le Moteur Symbolique l'évalue par une fonction booléenne $V \in \{0, 1\}$.
   Si $V = 0$ (Règle violée), $\mathcal{L}_{step} = \beta$ (où $\beta$ est massif, typiquement 1000). 
   **Impact :** Cela force l'abandon immédiat (backtrack) de l'arbre de recherche. L'IA apprend qu'il est infiniment plus coûteux de deviner faux que de prendre le temps de décomposer correctement la tâche.
2. **$\mathcal{L}_{sparse}$ (Régularisation L1)**
   $$ \mathcal{L}_{sparse} = \gamma \sum |v_i| $$
   **Impact :** Pousse l'écrasante majorité des dimensions à zéro. C'est l'exigence d'une pensée claire. Une "Masse" ne doit allumer que quelques neurones spécifiques de son segment $v_{ent}$, empêchant la mémoire "brouillée" et facilitant l'interprétabilité.
3. **$\mathcal{L}_{consist}$ (Loss Contrastive InfoNCE)**
   Garantit que la représentation audio de la pluie et la représentation visuelle de la pluie se projettent vers exactement le même vecteur. Permet l'Amodalité complète.

## 2.3 MCTS Latent (Monte Carlo Tree Search)
Pendant le "Mode Sommeil" (Consolidation Offline), le modèle utilise le MCTS dans l'espace latent.
*   **Sélection :** Parcourt l'arbre de pensée latent.
*   **Expansion :** Propose une nouvelle règle.
*   **Simulation :** Unroll de la boucle LSRA sur le port 26415.
*   **Rétropropagation :** Si la règle permet de résoudre la surprise bayésienne (diminuer l'erreur sur un cas d'échec stocké), le nœud est mis à jour et validé.
