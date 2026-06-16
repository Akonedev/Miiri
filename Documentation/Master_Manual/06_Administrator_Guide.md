# Chapitre 6 : Guide de l'Administrateur Système (MLOps)

L'administration du modèle OCM-26400 repose sur l'orchestration des micro-services via ZeroMQ et Docker.

## 6.1 Déploiement et Sécurité
- Le système ne doit jamais exposer les ports IPC 26400-26420 à l'extérieur.
- Déploiement via `docker-compose` avec un bridge réseau isolé (ex: `ocm-net`).
- **Allocations Matérielles :**
  - Ports 26401-26404 (Encodeurs) : Peuvent utiliser des Tensor Cores plus petits.
  - Port 26410 (Working Memory) : Exige la RAM système la plus rapide (HBM).
  - Ports 26415-26418 (Reasoning Core) : Ce sont les goulots d'étranglement du Test-Time Compute. Dédier l'essentiel de la puissance GPU (H100) ici.

## 6.2 Gestion du Mode Sommeil (Sleep Consolidation)
L'administrateur doit configurer un `cron` ou un Event-Trigger nocturne.
- Le Global Workspace (26400) est mis en mode `CONSOLIDATION`.
- L'IA fouille la Mémoire Épisodique (26411) pour trouver des erreurs.
- L'admin doit superviser que les nouvelles règles découvertes et transférées vers la Mémoire Sémantique (26412) ne créent pas de régressions.

## 6.3 OCM-Forge (L'Outil d'Automatisation)
OCM-Forge est l'outil CLI d'orchestration MLOps de bout-en-bout.
Usage basique :
`./ocm_forge.py --prompt "Crée un modèle de règles physiques" --modality text vision --max-iters 128`

Le script gère dynamiquement la création des vecteurs, le démarrage du réseau ZMQ, l'entraînement RL (Grokking) et l'exportation des poids finaux au format PyTorch (`.pt`).
