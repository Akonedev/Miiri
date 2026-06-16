# Chapitre 9 : Le Système "Vivant" (Living AGI), Orchestration Multi-Agents et Usage d'Outils

*Mise à jour majeure suite à l'audit de conformité AGI.*
Ce chapitre détaille comment l'Miiri passe d'un simple modèle statique à un **organisme cognitif vivant**, évolutif, et capable d'interagir physiquement avec le monde numérique (Browser/Computer Use). L'axiome fondamental reste : **Aucune architecture Frankenstein. Tout est nativement unifié dans le Mentalese.**

## 9.1 L'Évolution Dynamique (Dynamic Sizing & Living Capacity)
Les modèles classiques ont une taille fixe (ex: 70 Milliards de paramètres). L'Miiri introduit la **Croissance Paramétrique Dynamique**.
*   **Le Principe :** Le modèle n'a pas de taille fixe. Au fur et à mesure qu'il apprend en continu, s'il détecte que son espace latent ($d=256$) ou son cluster de récurrence (LSRA) sature (interférences entre les règles causales), l'Orchestrateur (Port 26400) **alloue automatiquement de nouveaux experts (Mixture of Experts - MoE) à la volée**, tant que les ressources matérielles (GPU/RAM) le permettent.
*   **Résultat :** C'est un modèle "vivant". Il grandit physiquement en fonction de ce qu'il ingère, comme un réseau synaptique humain.

## 9.2 Conscience de Soi et Incertitude Épistémique
L'IA doit savoir ce qu'elle ne sait pas.
*   **Implémentation Mathématique :** La dimension `[255]` du vecteur QPLS est l'**Indice d'Incertitude Épistémique**.
*   Si le modèle est confronté à un concept totalement inédit, cet indice monte à 1.0. 
*   Au lieu de deviner (hallucination), l'état latent déclenche l'action `[TOOL_USE_SEARCH]` pour acquérir la donnée manquante.

## 9.3 L'Orchestration Multi-Agents Interne (Zéro Frankenstein)
Le modèle n'appelle pas des scripts externes via LangChain. Les "Agents" sont des **sous-espaces d'attention à l'intérieur du Reasoning Core (LSRA)**.
*   **Le Débat Interne :** Lors du Test-Time Compute (TTC), le modèle clone son état latent en plusieurs "Points de Vue" (Ex: Un Agent Critique Logique, Un Agent Créatif). 
*   Ces vecteurs bouclent en parallèle. Avant de valider une étape, ils opèrent un produit scalaire (Cross-Attention) entre eux. S'il y a dissonance, ils se challengent jusqu'à converger vers un consensus mathématique validé par le Symbolic Gate (26412).
*   **Le Superviseur :** Le Global Workspace valide le travail des agents internes et force une rétroaction (Backtrack) si le consensus n'est pas "Production Grade".

## 9.4 Apprentissage Continu et Synthèse en Arrière-Plan
L'apprentissage n'est plus relégué au seul "Mode Sommeil".
*   **Real-Time Background Synthesis :** Un thread asynchrone (Non-Bloquant) tourne en permanence sur le Port 26400. 
*   Pendant que le modèle vous parle, ce thread observe la "Stream de Prompts", relie les nouvelles informations aux connaissances passées dans la Mémoire Épisodique (26411), et met à jour les poids des règles en temps réel.

## 9.5 L'Usage d'Outils Natif (Computer/Web Use)
Le modèle doit utiliser un navigateur, cliquer, scrapper, se connecter.
*   **La Solution Unifiée :** Les actions système ne sont pas des appels d'API externes. Elles font partie de l'**Omni-Tokenizer Universel** (Chapitre 8).
*   Le vocabulaire contient des tokens d'action purs : `[ACTION_CLICK_X_Y]`, `[ACTION_TYPE]`, `[ACTION_SCROLL]`.
*   **Le Flux Causal :**
    1. Le modèle constate son ignorance (Incertitude = 1.0).
    2. Le raisonnement latent (TTC) déduit que la solution est sur le Web.
    3. L'Omni-Decoder projette le vecteur sur les tokens `[ACTION_OPEN_BROWSER]`, `[ACTION_NAVIGATE]`.
    4. Le flux visuel de l'écran entre par le Lobe Visuel (26403). Le modèle "voit" la page web en même temps qu'il lit le DOM (HTML via le Lobe Texte 26401).
    5. Appariement multimodal (Scraping/Analyse profonde) : Les données sont digérées, le modèle apprend en live, met à jour son Mentalese, et vous répond.
    6. Tout ceci est protégé par des **Règles de Sécurité** strictes intégrées au Dictionnaire Sémantique (Port 26412), empêchant le modèle de cliquer sur des éléments destructeurs.
