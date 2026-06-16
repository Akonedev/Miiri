# Architecture OCM-26400 (Omni-Cognitive Mentalese)

## 1. Vision et Objectifs
Le projet OCM-26400 vise à créer une architecture IA de nouvelle génération qui ne repose plus sur la prédiction probabiliste (devinette) mais sur la **restitution causale par application de règles**.

### Principes Fondamentaux
- **Mentalese (Langage de la Pensée) :** Un espace latent explicite ($d=256$) et amodal.
- **Compositionnalité Garantie :** Utilisation d'opérateurs explicites pour combiner des primitives.
- **Grokking vs Scale :** La compétence émerge de la compréhension des bases (grokking) et de la décomposition des tâches, pas de l'augmentation des paramètres.
- **Efficient Reasoning :** Découplage de la profondeur de raisonnement du nombre de paramètres via la récurrence fenêtrée.

## 2. Topologie Réseau (Ports 26400-26420)
L'IA est structurée en lobes communicants :
- **26400 : Superviseur (Global Workspace)** - Conscience et Attention.
- **26401-26404 : Encodeurs Sensoriels** (Texte, Audio, Vision, 3D).
- **26410-26412 : Systèmes de Mémoire** (Scratchpad, Épisodique, Sémantique/Dictionnaire).
- **26415-26418 : Reasoning Core** - Cluster de récurrence (Test-Time Compute).

## 3. Le Vecteur Mentalese ($d=256$)
Partitionné pour l'interprétabilité :
- `[0-63]` : Entités et Radicaux.
- `[64-127]` : Modificateurs et Affixes.
- `[128-191]` : Opérateurs Causaux.
- `[192-255]` : Métadonnées d'État.

## 4. Algorithmes de Contrôle
- **Loss de Rigueur Causale :** Pénalise l'ambiguïté et le saut d'étape.
- **Surprise Bayésienne :** Déclenche l'apprentissage quand la prédiction du monde échoue.
- **Rétrospective Offline :** Cycle de "sommeil" pour consolider les expériences en règles sémantiques.
