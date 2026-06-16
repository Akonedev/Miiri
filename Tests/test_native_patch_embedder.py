import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.data.native_patch_embedder import NativePixelPatchEmbedder

def test_native_patch_embedder():
    """
    Test la conversion de pixels bruts vers l'espace Mentalese QPLS.
    Vérifie l'Orthogonal Sparsity (Règle d'Ingénierie Stricte).
    """
    batch_size = 2
    channels = 3
    image_size = 224
    patch_size = 16
    d_model = 256
    
    # Simule un tenseur d'image pure (ex: depuis OpenCV ou PIL)
    raw_pixels = torch.randn(batch_size, channels, image_size, image_size)
    
    embedder = NativePixelPatchEmbedder(
        image_size=image_size, 
        patch_size=patch_size, 
        in_channels=channels, 
        d_model=d_model
    )
    
    # Exécution de l'encodage
    qpls_vectors = embedder(raw_pixels)
    
    # 1. Vérification des dimensions
    num_patches = (image_size // patch_size) ** 2 # (224/16)^2 = 196
    assert qpls_vectors.shape == (batch_size, num_patches, d_model), "Mauvaise dimension de projection"
    
    # 2. Vérification de l'Orthogonal Sparsity (Anti-Mocking Rule)
    # Les opérateurs causaux (D128 à D191) doivent être STRICTEMENT à 0.0
    operators_segment = qpls_vectors[..., 128:192]
    assert torch.all(operators_segment == 0.0), "FATAL: Les pixels polluent l'espace des opérateurs causaux !"
    
    # 3. Vérification du tag de modalité (D192 = 2.0 pour la vision)
    metadata_segment = qpls_vectors[..., 192]
    assert torch.all(metadata_segment == 2.0), "Le tag de modalité visuelle n'est pas appliqué."
    
    print("\n[SUCCESS] Native Patch Embedder : Conforme au Mentalese QPLS et Orthogonal Sparsity validée.")

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
