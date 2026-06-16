# Deep Architecture Blueprint : Miiri
## Architecture Miiri (Pensée Unifiée) via Latent-to-Symbolic Recurrence

Ce document détaille l'architecture système, les flux de données et la topologie réseau de l'IA **Miiri**. Conçue pour une rigueur de niveau "Production Grade", cette architecture résout le problème des hallucinations des LLMs en remplaçant la génération autoregressive par un **Raisonnement Latent Vérifié**.

---

### 1. Vue Macro de l'Architecture (Topologie)

L'architecture est structurée autour d'un bus de messages à latence ultra-faible (ZeroMQ) sur la plage de ports **26400-26420**. Elle sépare strictement la *Perception*, la *Mémoire*, et le *Raisonnement*.

```mermaid
graph TD
    %% Couche Sensorielle (Perception)
    subgraph "Sensory Lobes (Amodal Fusion)"
        L[Linguistic Port 26401]
        A[Audio Port 26402]
        V[Vision Port 26403]
        S[Spatial 3D Port 26404]
    end

    %% Superviseur (Global Workspace)
    GW((Global Workspace<br>Port 26400<br>Attention & Surprise))

    %% Mémoire
    subgraph "Cognitive Memory"
        WM[(Working Memory / Scratchpad<br>Port 26410)]
        EM[(Episodic Memory<br>Port 26411)]
        SM[(Semantic Memory / Dictionary<br>Port 26412)]
    end

    %% Raisonnement (Test-Time Compute)
    subgraph "Recurrent Reasoning Core"
        RC1[Node 26415]
        RC2[Node 26416]
        RC3[Node 26417]
        RC4[Node 26418]
    end

    %% Action
    OUT[Action / Output<br>Ports 26419-26420]

    %% Flux de données
    L & A & V & S -->|Projection Amodale| GW
    GW <-->|Écriture/Lecture| WM
    GW <-->|Rétrospective| EM
    GW <-->|Validation Symbolique| SM
    
    WM -->|État Latent d=256| RC1
    RC1 -->|Boucle Latente| RC2
    RC2 -->|Boucle Latente| RC3
    RC3 -->|Boucle Latente| RC4
    RC4 -.->|Backtrack si erreur| WM
    RC4 -->|Convergence = 1.0| OUT

    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef mem fill:#9cf,stroke:#333,stroke-width:2px;
    class GW core;
    class WM,EM,SM mem;
```

---

### 2. Vue Micro : L'Espace Latent Quadri-Partitionné (QPLS)

**Innovation Baptisée : Quad-Partitioned Latent Semantic Space (QPLS)**
Contrairement aux espaces latents continus classiques (où le sens est distribué uniformément, rendant la composition mathématiquement instable), le vecteur Mentalese $d=256$ du Miiri est **strictement partitionné**. 

Cette innovation garantit que les opérations (ex: "ajouter un pluriel", "appliquer la gravité") n'altèrent pas l'identité de l'objet manipulé.

```mermaid
erDiagram
    Mentalese_Vector_d256 {
        float Dimensions_0_63 "Entities & Roots (Quoi ?)"
        float Dimensions_64_127 "Properties & Affixes (Comment ?)"
        float Dimensions_128_191 "Causal Operators (Lien ?)"
        float Dimensions_192_255 "State Metadata (Statut ?)"
    }
```

#### Anatomie d'une Pensée : `COMPOSE(Chant, Pluriel)`
1. **D[0-63] (Entité) :** Le radical `chant-` active un sous-vecteur spécifique. *Exemple: [0.9, -0.2, ...]*
2. **D[64-127] (Propriété) :** L'affixe de pluralité `-s` active une autre zone. *Exemple: [0.0, 0.8, ...]*
3. **D[128-191] (Opérateur) :** L'opérateur d'assemblage `CONCAT` est actif.
4. **D[192-255] (Métadonnées) :** Le vecteur enregistre `Confiance: 0.99`, `Source: Port 26401`.

**Pourquoi ça marche ?** Si le modèle essaie d'appliquer la règle `[Gravité]` (D128-191) à `[La notion de Joie]` (D0-63), le Moteur Symbolique (Port 26412) lit le vecteur. Il voit une incompatibilité de "Typage Sémantique" et refuse la composition. **L'hallucination est tuée au niveau matriciel.**

---

### 3. Le Moteur de Récurrence Fenêtrée : LSRA

**Innovation Baptisée : Latent-to-Symbolic Recurrent Architecture (LSRA)**
Le plus grand défaut des Transformers (GPT-4, Llama) est que la profondeur de leur réflexion est égale à leur nombre de couches (Paramètres). 
Le **LSRA** casse ce mur.

```mermaid
sequenceDiagram
    participant WM as Scratchpad (Port 26410)
    participant RC as Reasoning Core (26415)
    participant SM as Semantic Memory (26412)
    
    Note over WM, SM: Début du cycle de raisonnement
    WM->>RC: Envoi État Latent Initial (t=0)
    
    loop Test-Time Compute (Boucle jusqu'à Convergence)
        RC->>RC: Transformation Non-Linéaire (Couche Transformer partagée)
        RC->>SM: Requête de Validation Symbolique (L'étape est-elle légale ?)
        
        alt Règle Causale Violée
            SM-->>RC: Erreur (Reward = -1000)
            RC->>WM: Backtrack (Restauration état t-1)
        else Étape Légale
            SM-->>RC: Validé (Reward = +1)
            RC->>RC: Mise à jour État Latent (t = t+1)
        end
    end
    
    RC->>WM: État Final Convergé (Confiance > 0.95)
```

**Pourquoi est-ce "Production Grade" ?**
Au lieu de forcer un LLM de 70 Milliards de paramètres à tout calculer en un passage (ce qui coûte énormément de VRAM et produit des erreurs), Miiri utilise un modèle très léger (ex: 1 Milliard de paramètres) mais le fait boucler 64 fois dans sa RAM locale. 
Le coût de calcul est basculé vers le moment de l'inférence (**Test-Time Compute**), imitant le système "Slow Thinking" (Système 2) du cerveau humain.
