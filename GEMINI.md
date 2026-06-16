# Project Rules: Miiri Architecture

## Strict Engineering Constraints
1. **No Mocking/Faking Data:** The data pipelines must ingest REAL linguistic or physical datasets. Do not use `torch.zeros()` or mock semantic dictionaries.
2. **True Benchmarks Only:** Benchmarking scripts must run actual matrix multiplications or Triton kernels. Do not simulate time using `time.sleep()`.
3. **Hardware Truth:** Acknowledge local hardware limits. Build the code for distribution on real clusters (e.g., PyTorch DDP, DeepSpeed) and test unitarily.
4. **Architectural Focus:** Build single, mathematically sound components.
5. **Sequential Task Decomposition (Rule 8):** When given a list of tasks, treat it as a global backlog. Do not implement everything in one prompt. Decompose, plan, and execute one by one with the optimal method.
6. **Real & Optimized Implementation (Rule 9):** Code must be hardware-optimized for the target system (e.g., Triton kernels for GPU bottleneck bypass). No naive Python loops for performance-critical paths.
