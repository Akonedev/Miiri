# Mathematical Formulations and Loss Functions

This document rigorously defines the mathematical framework underlying the **Architecture Miiri (Pensée Unifiée) (Miiri-256)** architecture, specifically detailing the **Amodal Mentalese Vector (AMV-256)** and the **Causal Rigor Loss ($\mathcal{L}_{CausalRigor}$)**.

## 1. The Amodal Mentalese Vector (AMV-256)

Let $v \in \mathbb{R}^{256}$ be the latent state representing a single thought or concept within the cognitive workspace. Unlike standard continuous embeddings where semantic meaning is distributed globally, the AMV-256 is subject to a strict hard-partitioning constraint:

$$ v = [v_{ent} \parallel v_{prop} \parallel v_{op} \parallel v_{meta}] $$

Where:
- $v_{ent} \in \mathbb{R}^{64}$ encodes the entity or root primitive (e.g., *Cat*, *Mass*).
- $v_{prop} \in \mathbb{R}^{64}$ encodes modifiers and properties (e.g., *Plural*, *Velocity*).
- $v_{op} \in \mathbb{R}^{64}$ encodes causal or syntactic operators (e.g., $\oplus_{compose}$, $\otimes_{derive}$).
- $v_{meta} \in \mathbb{R}^{64}$ encodes metadata (Confidence $c \in [0,1]$, Modality Source).

### Amodal Projection
Given inputs from $N$ different modalities (Text $T$, Audio $A$, Vision $V$, Spatial $S$), let $f_i$ be the modality-specific encoder. The amodal alignment constraint enforces that for a given concept $C$:

$$ f_T(C) \approx f_A(C) \approx f_V(C) \approx f_S(C) \approx v_C $$

This is enforced via an InfoNCE contrastive loss over the modality pairs.

---

## 2. Causal Rigor Loss ($\mathcal{L}_{CausalRigor}$)

**Innovation Baptisée : Amodal Consistency & Step Penalty (ACSP)**

To eliminate "guessing" (statistical hallucination) and enforce strict rule-based reasoning, the training objective does not merely supervise the final output $y$, but the trajectory of latent steps $v^{(1)}, v^{(2)}, \dots, v^{(T)}$.

The total loss is defined as:

$$ \mathcal{L}_{CausalRigor} = \alpha \mathcal{L}_{align} + \beta \mathcal{L}_{step} + \gamma \mathcal{L}_{sparse} + \delta \mathcal{L}_{consist} $$

### 2.1 Alignment to the Primitive Dictionary ($\mathcal{L}_{align}$)
Let $\mathcal{D}$ be the semantic dictionary of grokked primitives stored on Port 26412. Every proposed latent state $v^{(t)}$ must align with a valid primitive or a legally composed set of primitives.

$$ \mathcal{L}_{align} = \frac{1}{T} \sum_{t=1}^{T} \min_{d \in \mathcal{D}} \left( 1 - \frac{v^{(t)} \cdot d}{\|v^{(t)}\| \|d\|} \right) $$

This penalizes the model heavily if it generates an "ambiguous" vector that does not correspond to a known concept or legal combination.

### 2.2 Step Validity and Backtracking Penalty ($\mathcal{L}_{step}$)
Let $V(\cdot, \cdot, \cdot)$ be the symbolic verification function (Port 26412) that takes an entity $v_{ent}$, a property $v_{prop}$, and an operator $v_{op}$. $V$ returns $1$ if the operation is legal (e.g., $Position = \frac{1}{2}gt^2$), and $0$ otherwise.

$$ \mathcal{L}_{step} = \sum_{t=1}^{T} \begin{cases} 
      0 & \text{if } V(v^{(t)}_{ent}, v^{(t)}_{prop}, v^{(t)}_{op}) = 1 \\
      P_{backtrack} & \text{if } V(v^{(t)}_{ent}, v^{(t)}_{prop}, v^{(t)}_{op}) = 0 
   \end{cases} $$

Where $P_{backtrack} \gg 1$ (e.g., 1000.0). This massive penalty forces the model to use the **Working Memory (Scratchpad)** to decompose problems, as attempting to jump directly to a final answer without valid intermediate primitives will almost certainly violate the step check.

### 2.3 Sparsity Constraint ($\mathcal{L}_{sparse}$)
To prevent the model from using the 256 dimensions to memorize noise, an L1 regularization is applied to ensure that only the strictly necessary dimensions of $v$ are active for any given primitive:

$$ \mathcal{L}_{sparse} = \lambda \sum_{i=1}^{256} |v_i| $$

### 2.4 Multimodal Consistency ($\mathcal{L}_{consist}$)
Using the InfoNCE loss for a batch of $N$ positive cross-modal pairs $(v_i, u_i)$ representing the same concept:

$$ \mathcal{L}_{consist} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(v_i \cdot u_i / \tau)}{\sum_{j=1}^{N} \exp(v_i \cdot u_j / \tau)} $$

---

## 3. Test-Time Compute: The Latent MCTS

During inference, the **Latent-to-Symbolic Recurrent Architecture (LSRA)** unrolls the network dynamically. Instead of generating tokens, the model updates its internal state $v^{(t)}$.

$$ v^{(t+1)} = \text{TransformerBlock}(v^{(t)}, \text{Context}_{1..t}) $$

The iteration stops at $t = T^*$ when the Confidence metadata dimension surpasses the Grokking Threshold $\tau_{grok}$:

$$ T^* = \min \{ t \mid v^{(t)}_{meta\_confidence} \ge \tau_{grok} \} $$

If $T^* > \text{MaxIterations}$ (e.g., 64), the model triggers an `[ANOMALIE_CAUSALE]` event, sending the problem to the Episodic Memory for offline consolidation (Sleep Mode).
