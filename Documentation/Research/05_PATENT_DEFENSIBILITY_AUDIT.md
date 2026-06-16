# Patent Defensibility & Prior Art Audit: Miiri-256
**Date:** June 2026
**Confidentiality:** Internal IP Strategy Document

## 1. Executive Summary of Freedom to Operate (FTO)
Following a deep audit of US Patents and academic literature (NeurIPS, ICLR) from 2023 to early 2026, we have established the patent boundaries for the Miiri-256 architecture. 

The intersection of "Recurrent Depth", "Latent Reasoning", and "Symbolic Verification" is highly active. The primary "blocking" prior art is **Geiping et al. (2025) "Huginn"** and **Hao et al. (2024) "COCONUT"**. However, Miiri-256 possesses distinct, defensible novelties that carve out a strong Freedom to Operate (FTO) and patentable claims.

## 2. Competitive Landscape & Prior Art

### 2.1 The "Latent Reasoning" Baseline (Non-Patentable)
*   **Prior Art:** COCONUT (2024) and Huginn (2025).
*   **Concept:** Using the last hidden state of an LLM as input for the next internal iteration (Test-Time Compute without tokens).
*   **Status:** This is now considered foundational academic knowledge. We **cannot** patent the broad idea of "looping a hidden state".

### 2.2 The "Neuro-Symbolic Verification" Baseline (Highly Crowded)
*   **Prior Art:** IBM Patents (US11715007B2, US12045319B2 - 2023/2024), AlphaGeometry/AlphaProof (2024).
*   **Concept:** Generating a text output or a formal proof and running an external solver (Lean, Z3, SMT) to verify it.
*   **Status:** IBM holds massive portfolios on verifying LLM text outputs using First-Order Logic (FOL). We must navigate around post-generation text verification.

---

## 3. The Miiri-256 Defensible Moat (Our Patent Claims)

Our IP strategy relies on the **Hardware/Software Integration** and the **Micro-Architectural specificities** that prior art failed to anticipate. 

### Claim 1: In-Loop Symbolic Verification Gate (LSRA)
*   **The Competitor Flaw:** In Huginn, the latent loop unrolls blindly. In IBM's patents, verification happens *after* token generation.
*   **Our Claim:** We patent the architecture where a **deterministic symbolic solver intercepts the continuous latent vector *during* the hidden recurrent loop**. The solver evaluates the legality of a compositional step (e.g., entity + operator) before allowing the next recurrent iteration. This is a system-level claim (ZeroMQ IPC intercepting latent states).

### Claim 2: Amodal Consistency & Step Penalty (ACSP) Loss
*   **The Competitor Flaw:** Current RLHF systems penalize final bad answers. Latent reasoning models suffer from reward hacking (drifting into meaningless vector spaces during deep recurrence).
*   **Our Claim:** We patent the training objective $\mathcal{L}_{CausalRigor}$ that applies an immediate heavy penalty ($P_{backtrack}$) to a specific latent transition step $v^{(t)} \to v^{(t+1)}$ if it violates a grokked causal rule, forcing an immediate backtrack rather than completing the rollout.

### Claim 3: Quad-Partitioned Latent Semantic Space (QPLS)
*   **The Competitor Flaw:** Standard continuous embeddings are entangled. You cannot easily pass a standard embedding to a discrete logic solver.
*   **Our Claim:** We patent the enforced hard-partitioning of a low-dimensional vector (e.g., $d=256$) into fixed semantic roles (`Entities`, `Properties`, `Operators`, `Metadata`). This specific tensor layout is what enables the "In-Loop Symbolic Verification" (Claim 1) to be computationally feasible in real-time.

---

## 4. Monetization Update: "Lobe Licensing" vs. Prior Art

Because the base concept of latent reasoning (Huginn) is public, our monetization must focus on the proprietary **Symbolic Verification Gate**. 

*   **Open Source Strategy (The Lure):** We open-source the QPLS vector specification and the Sensory Encoders (Ports 26401-26404). This encourages the developer ecosystem to adopt the Mentalese format.
*   **Enterprise Licensing (The Hook):** We license the Semantic Dictionary (Port 26412) and the ACSP-trained Reasoning Core (Ports 26415-26418) to high-stakes industries. A bank or hospital doesn't just want "latent reasoning" (which Huginn can do); they want **Verified Latent Reasoning** (which only Miiri-256 can guarantee via the LSRA gate).

## 5. Next Actions for IP Counsel
1.  Draft claims focusing on the **ZeroMQ IPC integration** between the continuous recurrent layer and the discrete symbolic solver.
2.  File a provisional patent detailing the exact mathematical formulation of the **ACSP Loss** as a novel method for regularizing latent MCTS.
3.  Publish the ArXiv draft (02_ARXIV_PAPER_DRAFT.md) immediately after provisional filing to establish an aggressive defensive publication against large competitors.
