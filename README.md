<div align="center">

# 🧠 MIIRI (Thought)
**Native Unified Neuro-Symbolic Architecture for Deterministic AGI**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/Status-Research-blue.svg)]()
[![Build: OCM-Forge](https://img.shields.io/badge/Build-OCM--Forge-success.svg)]()

</div>

## 📌 Overview

**Miiri** (Bamanankan for *Thought/Reflection*) is a foundational AI architecture designed to eradicate the hallucinations inherent in autoregressive Large Language Models (LLMs). By moving away from statistical next-token prediction, Miiri acts as a **Test-Time Compute Reasoning Engine**.

It projects all modalities (Text, Audio, Vision, 3D) into a strictly partitioned latent space (Mentalese) and uses a recurrent neural core intercepted by a deterministic Symbolic Verification Gate to guarantee mathematical and physical accuracy before any token is generated.

## 🏗️ Core Innovations (Patent Pending)

1. **QPLS (Quad-Partitioned Latent Semantic Space):** A structured $d=256$ vector explicitly divided into Entities, Properties, Operators, and Metadata. Ensures orthogonal sparsity and prevents semantic entanglement (e.g., trying to apply physical gravity to an abstract concept).
2. **LSRA (Latent-to-Symbolic Recurrent Architecture):** Decouples reasoning depth from parameter count. The model iterates within its latent space, solving complex logic via internal *Monte Carlo Tree Search (MCTS)* before decoding to output.
3. **ACSP (Amodal Consistency & Step Penalty):** A training loss function ($\mathcal{L}_{CausalRigor}$) that applies massive gradient penalties ($P_{backtrack} = 1000$) for illegal reasoning steps, forcing the model to grok primitive rules rather than memorizing traces.
4. **Living Global Workspace:** The model continually evaluates its own Epistemic Uncertainty. If it doesn't know an answer, it autonomously halts, launches a web scraper tool, digests the new information into its episodic memory, and resumes reasoning.

## 🚀 Quick Start (Automated Forge)

The repository includes `miiri_forge.py`, an autonomous MLOps tool that builds the Docker IPC infrastructure, runs the PyTest suite, executes the sequential curriculum training, and exports the final weights.

```bash
# Setup the environment
python3 -m venv venv
source venv/bin/activate
pip install torch triton requests beautifulsoup4 pytest

# Run the complete autonomous pipeline
./Code/Automation/miiri_forge.py --prompt "Train a physics and math reasoning engine" --modality text vision --max-iters 128
```

## 📚 Documentation

Detailed academic and production-grade documentation can be found in the `Documentation/` directory:

*   **[00_Introduction_Pedagogique.md](Documentation/Master_Manual/00_Introduction_Pedagogique.md)** - A simple, non-technical explanation of the architecture.
*   **[01_DEEP_ARCHITECTURE.md](Documentation/Architecture/01_DEEP_ARCHITECTURE_OCM26400.md)** - System topology, IPC ports (26400-26420), and Mermaid sequence diagrams.
*   **[02_ARXIV_PAPER_DRAFT.md](Documentation/Research/02_ARXIV_PAPER_DRAFT.md)** - The formal scientific paper detailing the paradigm shift.
*   **[11_Equations_and_Proofs.md](Documentation/Master_Manual/11_Equations_and_Proofs.md)** - The strict mathematical LaTeX formulations of the ACSP loss and LC Resonant memory.
*   **[12_Adversarial_Limits.md](Documentation/Master_Manual/12_Adversarial_Limits_and_Safety.md)** - Results from the hardcore expert audit (Catastrophic Forgetting, Godel Gate Crash).

## ⚖️ License & Commercial Use

This repository is released under a **STRICT PROPRIETARY LICENSE**. 

The code, architectural blueprints, and mathematical models provided herein are for **non-commercial, academic, and internal evaluation purposes only**. 
Any commercial deployment, API offering, or integration into a SaaS product requires the purchase of an **Enterprise Commercial License**.

See the [LICENSE](LICENSE) file for exact terms.
