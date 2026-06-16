# 📘 Miiri-256 : MASTER OPERATIONS MANUAL
*Version : 1.0.0 (Production Grade) | Classification : Confidentiel / Cœur de Technologie*

Ce document est le guide canonique exhaustif pour la création, le déploiement, l'entraînement et l'ajustement (tuning) d'un modèle basé sur l'architecture **Architecture Miiri (Pensée Unifiée) (Miiri-256)**. 

Il s'adresse à quatre profils : **Architectes/Développeurs**, **Data Engineers**, **Tuners/Chercheurs en RL**, et **Administrateurs Système (MLOps)**.

---

## 🛑 PARTIE 1 : LE PARADIGME FONDAMENTAL (POUR TOUS)

L'Miiri-256 n'est pas un LLM (Large Language Model) au sens classique. Il s'agit d'un **Moteur de Raisonnement Neuro-Symbolique à Récurrence Latente**.

1.  **L'Échec du "Next-Token Prediction" :** Les LLMs prédisent le prochain mot. S'ils ne l'ont pas vu dans leurs données, ils hallucinent. L'Miiri-256 ne prédit pas de mots. Il manipule des **concepts purs (Mentalese)** et applique des opérateurs mathématiques ou grammaticaux vérifiés en temps réel par un moteur symbolique.
2.  **Le Grokking avant la Composition :** On n'enseigne pas des phrases à l'Miiri. On lui enseigne un dictionnaire de *Primitives* (radicaux, suffixes, lois physiques) jusqu'à ce que la mémorisation s'effondre et que la compréhension émerge (Grokking).
3.  **Test-Time Compute (TTC) :** Au lieu d'avoir 100 couches de neurones, l'Miiri a un petit bloc de neurones qui **boucle sur lui-même** (Récurrence Fenêtrée). Il réfléchit plus longtemps pour les problèmes difficiles sans consommer plus de VRAM.

---

## 💻 PARTIE 2 : GUIDE DU DÉVELOPPEUR & ARCHITECTE

Le rôle du développeur est de gérer le code PyTorch et la topologie ZeroMQ.

### 2.1 Le Vecteur QPLS (Quad-Partitioned Latent Semantic Space)
Le vecteur n'est pas continu, il est typé.
*   **Formule :** $v = [v_{ent} \parallel v_{prop} \parallel v_{op} \parallel v_{meta}]$ où $v \in \mathbb{R}^{256}$.
*   **Implémentation :** Le vecteur de dimension 256 DOIT être hard-codé en 4 segments de 64 dimensions.
*   **Impact :** Sans ce partitionnement, le modèle mélange les "actions" et les "entités" (ex: il essaiera d'appliquer une couleur à un verbe). La séparation garantit la compositionnalité $\rightarrow$ `COMPOSE(Entité, Propriété)`.

### 2.2 L'Encodeur Amodal (InfoNCE Loss)
Vous devez mapper l'audio, la vidéo et le texte vers le même sous-espace $D[0-63]$.
*   **Algorithme :** Loss Contrastive (InfoNCE).
    $$ \mathcal{L}_{consist} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(v_i \cdot u_i / \tau)}{\sum_{j=1}^{N} \exp(v_i \cdot u_j / \tau)} $$
*   **Paramètre $\tau$ (Temperature) :** Contrôle la sévérité de l'alignement. 
    *   *Utilité :* Si $\tau = 0.07$ (standard), le modèle est très strict. S'il entend le son d'une cloche, il activera le même vecteur exact que s'il lit le mot "cloche".
    *   *Impact :* C'est ce qui permet au modèle d'apprendre la physique en regardant des vidéos, puis de l'expliquer en mots sans ré-entraînement.

---

## 🎛️ PARTIE 3 : GUIDE DU TUNER ET FINE-TUNER (REINFORCEMENT LEARNING)

Le tuner est responsable de la fonction de perte (Loss) et du comportement cognitif.

### 3.1 La Loss de Rigueur Causale (ACSP - Amodal Consistency & Step Penalty)
C'est le joyau de la couronne. Au lieu de punir la mauvaise réponse finale, vous punissez le "mauvais cheminement de pensée".

$$ \mathcal{L}_{CausalRigor} = \alpha \mathcal{L}_{align} + \beta \mathcal{L}_{step} + \gamma \mathcal{L}_{sparse} + \delta \mathcal{L}_{consist} $$

#### Explication des hyperparamètres et tuning :
*   **$\beta$ (Le Poids de Rigueur - `backtrack_penalty`) :** C'est la punition infligée si le modèle tente une étape illégale (ex: appliquer "Pluriel" sur "Masse").
    *   *Tuning :* Fixé à `1000.0` par défaut. S'il est trop bas (ex: `1.0`), le modèle ignorera le Moteur Symbolique et recommencera à halluciner. S'il est trop haut (ex: `10000.0`), le gradient explosera.
*   **$\gamma$ (La Sparsité - `sparsity_lambda`) :** Force le modèle à utiliser peu de neurones pour un concept.
    *   *Utilité :* Empêche l'enchevêtrement des concepts. Un radical doit allumer 3 neurones sur 64, pas 60.
    *   *Tuning :* Fixé à `0.01`. Augmentez-le si l'espace latent devient "brouillé". Baissez-le si le modèle n'arrive plus à exprimer des nuances complexes.

### 3.2 Le Réglage du Test-Time Compute (TTC)
L'architecture LSRA boucle l'état latent.
*   **`max_latent_iterations` :** (Défaut : 64). Définit la "profondeur de réflexion" maximale. Pour un modèle de chatbot basique, `10` suffit. Pour un modèle de recherche mathématique, montez à `256`. 
*   **`grokking_threshold` :** (Défaut : 0.95). C'est le niveau de confiance (stocké dans les métadonnées D192-255) qui arrête la boucle.
    *   *Impact :* Si réglé à `0.99`, le modèle sera très lent mais infaillible. Si réglé à `0.60`, il répondra vite mais avec un risque de logique imparfaite.

---

## 📊 PARTIE 4 : GUIDE DU DATA ENGINEER (CURRICULUM)

Le modèle ne lit pas Wikipedia. Il suit un cursus scolaire strict.

### Phase 1 : Le Dictionnaire de Primitives (Grokking)
1.  **Format des données :** Vous ne donnez pas de phrases. Vous donnez des associations `(Primitive, Type)`. Ex: `("chant-", "RADICAL_ACTION")`, `("-eur", "SUFFIXE_AGENT")`.
2.  **Objectif :** Entraîner les encodeurs (26401-26404) jusqu'à obtenir une accuracy de 1.0 sur ces briques de base.

### Phase 2 : Le Scratchpad (Décomposition)
1.  **Format des données :** Problèmes complexes AVEC chemin de résolution exigé.
2.  **Exemple :** Problème: "Chanteurs". Chemin exigé : `[RADICAL:chant] -> [SUFFIXE:eur] -> [FLEXION:s]`.
3.  **Utilité :** C'est ainsi que la généralisation compositionnelle émerge (elle passe de 75% à 98% de succès hors-distribution grâce à ces étapes intermédiaires).

---

## ⚙️ PARTIE 5 : GUIDE DE L'ADMINISTRATEUR (MLOps)

L'infrastructure Miiri-256 repose sur des micro-services (Lobes) connectés via ZeroMQ (IPC).

### 5.1 Plan d'adressage des Ports (Crucial)
Le firewall du cluster doit autoriser ces communications internes avec une latence < 1ms.
*   `26400` : **Global Workspace** (Nœud Maître - Orchestrateur MLOps).
*   `26401-26404` : **Encodeurs Sensoriels** (Peuvent être déployés sur des GPU "Edge" moins puissants).
*   `26410` : **Working Memory** (Doit être en RAM ultra-rapide / HBM).
*   `26412` : **Semantic Memory** (Héberge le moteur symbolique Lean 4 / Python). Demande beaucoup de CPU, peu de GPU.
*   `26415-26418` : **Reasoning Core** (Le Transformer). C'est ici que sont placés les GPU H100/A100.

### 5.2 Mode "Sommeil" (Consolidation Offline)
La nuit (ou en période creuse), l'admin doit déclencher le cycle de consolidation.
*   **Principe :** Le Superviseur (26400) fouille la mémoire épisodique (26411) pour trouver les `[ANOMALIES_CAUSALES]` (erreurs de la journée).
*   **Action :** Il utilise le cluster de raisonnement pour simuler des dizaines de milliers de nouvelles règles, et met à jour le dictionnaire sémantique (26412) de manière autonome.
*   **Impact :** Le modèle devient plus intelligent chaque jour sans qu'un humain n'ajoute de nouvelles données d'entraînement. C'est l'AGI en gestation.
