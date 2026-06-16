# Project Rules: Miiri Architecture

## Strict Engineering Constraints
1. **No Mocking/Faking Data:** The `train_miiri.py` and data pipelines must ingest REAL linguistic or physical datasets. Do not use `torch.zeros()` or mock semantic dictionaries to simulate grokking.
2. **True Benchmarks Only:** Benchmarking scripts (`test_ultimate_limits.py`, etc.) must run actual matrix multiplications or Triton kernels. Do not simulate time using `time.sleep()`.
3. **Hardware Truth:** Acknowledge local hardware limits. Do not claim a model is "fluent" or "fully trained" on an 8GB GPU. Build the code for distribution on real clusters (e.g., PyTorch DDP, DeepSpeed) and test unitarily.
4. **Architectural Focus:** Future work must focus on building single, mathematically sound components:
   - Writing the `Triton` kernel for the Latent-to-Symbolic Recurrence to bypass the Python PCIe bottleneck.
   - Writing the real PyTorch parser to map Wikipedia text to the 256-d QPLS vector.
   - Implementing the actual InfoNCE contrastive alignment loop with a real Vision Transformer (ViT) and Audio Spectrogram backbone.
