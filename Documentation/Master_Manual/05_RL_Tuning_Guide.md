# Chapitre 5 : Guide du Tuner et Hyperparamètres (Reinforcement Learning)

L'ajustement du Reinforcement Learning (RL) en post-training est ce qui différencie un modèle instable d'un modèle expert.

## 5.1 Les Hyperparamètres Critiques de la Loss ACSP

| Hyperparamètre | Rôle | Valeur Recommandée | Impact si mal réglé |
| :--- | :--- | :--- | :--- |
| `backtrack_penalty` ($\beta$) | Pénalité pour violation causale. | `1000.0` | Trop bas : le modèle recommence à halluciner. Trop haut : gradients instables. |
| `sparsity_lambda` ($\gamma$) | Pression L1 sur le vecteur latent. | `0.01` | Trop bas : espace latent bruité (entanglement). Trop haut : le modèle perd sa capacité d'expression. |
| `temperature` ($\tau$) | Sévérité de l'alignement InfoNCE. | `0.07` | Gère la fusion amodale. Trop bas : les vecteurs fusionnent trop durement et perdent leur nuance modale. |

## 5.2 Tuning du Scaling à l'Inférence
Le tuner a la responsabilité de régler les paramètres du "Test-Time Compute".

- **`max_latent_iterations` :** Nombre maximal de fois où le vecteur latent boucle dans le Reasoning Core (Ports 26415-18).
  - *Tuning Fin :* Augmentez ce chiffre si les problèmes nécessitent des analogies lointaines (ex: prouver un théorème complexe). Une valeur de `64` ou `128` est excellente pour de la Recherche, mais engendre une latence de réponse plus élevée.
- **`grokking_threshold` :** Le seuil de certitude du MCTS dans la dimension métadonnée.
  - *Tuning Fin :* À fixer à `0.95` ou `0.99` pour des opérations de "Production Grade" (Banques, Hôpitaux).

## 5.3 Transfert de Retour d'Expérience
Lors des phases d'erreurs récurrentes, le tuner ne doit pas ré-entraîner les primitives de base. Le système s'améliore via le transfert de connaissances :
Si le modèle échoue, le tuner vérifie les logs d'événements du Workspace Global (`[ANOMALIE_CAUSALE]`). Le RL scaling modifie la politique de choix des "chemins" dans l'arbre de décision latent.
