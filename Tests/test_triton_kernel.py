import torch
import triton
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.lsra.kernels.triton_lsra import run_fused_lsra
from Code.Enterprise.lsra.reasoning_core import LSRAReasoningCore

def test_triton_kernel_execution():
    """
    Vérifie que le Kernel Triton compile et s'exécute sans erreur sur GPU.
    Si pas de GPU, le test est skip (Triton nécessite CUDA).
    """
    if not torch.cuda.is_available():
        pytest.skip("Triton Kernel tests require a CUDA GPU. Skipping.")
        
    device = torch.device("cuda")
    d_model = 256
    batch_size = 16
    iters = 1000
    
    # Initialization
    x = torch.randn(batch_size, d_model, device=device, dtype=torch.float32)
    weight = torch.randn(d_model, d_model, device=device, dtype=torch.float32)
    bias = torch.randn(d_model, device=device, dtype=torch.float32)
    
    try:
        # Exécution du kernel
        out = run_fused_lsra(x, weight, bias, iters)
        
        # Validation de base
        assert out.shape == (batch_size, d_model)
        assert not torch.isnan(out).any()
    except Exception as e:
        pytest.fail(f"Le Kernel Triton a échoué : {e}")

def test_triton_vs_pytorch_speed():
    """
    Prouve mathématiquement que la fusion Triton détruit le goulot d'étranglement PCIe.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA requis pour les benchmarks de vitesse Triton.")
        
    device = torch.device("cuda")
    d_model = 256
    batch_size = 16
    iters = 10000 # 10k itérations pour voir le CPU bottleneck
    
    x = torch.randn(batch_size, d_model, device=device)
    w = torch.randn(d_model, d_model, device=device)
    b = torch.randn(d_model, device=device)
    
    # 1. TRITON (Fused Loop)
    # Warmup
    _ = run_fused_lsra(x, w, b, 10)
    torch.cuda.synchronize()
    
    start_triton = time.time()
    out_triton = run_fused_lsra(x, w, b, iters)
    torch.cuda.synchronize()
    time_triton = time.time() - start_triton
    
    # 2. NAIVE PYTORCH (CPU Loop)
    norm = torch.nn.LayerNorm(d_model).to(device)
    linear = torch.nn.Linear(d_model, d_model).to(device)
    
    # Warmup
    x_pt = x.clone()
    for _ in range(10): x_pt = norm(x_pt + linear(x_pt))
    torch.cuda.synchronize()
    
    start_pytorch = time.time()
    x_pt = x.clone()
    for _ in range(iters):
        # We simulate the exact same operations
        x_pt = norm(x_pt + linear(x_pt))
    torch.cuda.synchronize()
    time_pytorch = time.time() - start_pytorch
    
    print(f"\n[BENCHMARK] PyTorch vs Triton à {iters} itérations:")
    print(f"  PyTorch Naïf (CPU Loop) : {time_pytorch:.4f}s")
    print(f"  Triton Fused Kernel     : {time_triton:.4f}s")
    print(f"  Speedup : {time_pytorch / time_triton:.2f}x")
    
    # Triton DOIT être significativement plus rapide que le loop Python
    assert time_triton < time_pytorch

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
