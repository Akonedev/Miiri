import torch
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Code.Enterprise.lsra.omni_detokenizers import VisualDetokenizer, AudioDetokenizer, World3DDetokenizer
from Code.Enterprise.tools.mcp_executor import MCPExecutor

def test_visual_detokenizer():
    # Simulate a sequence of 196 thought tokens (e.g. 14x14 grid for a 224x224 image)
    batch_size = 2
    num_patches = 196 
    d_model = 256
    
    latent_sequence = torch.randn(batch_size, num_patches, d_model)
    decoder = VisualDetokenizer(d_model=d_model, out_channels=3, patch_size=16, image_size=224)
    
    pixels = decoder(latent_sequence)
    
    # Verify the output is a valid image tensor [Batch, Channels, Height, Width]
    assert pixels.shape == (batch_size, 3, 224, 224), "La projection visuelle a échoué. Mauvaises dimensions de tenseur."
    # Verify normalization
    assert torch.max(pixels) <= 1.0 and torch.min(pixels) >= 0.0, "Les pixels ne sont pas normalisés entre 0 et 1 (Sigmoid failure)."
    print("\n[PASS] Visual Detokenizer: Tenseur d'image 224x224 généré avec succès depuis le Mentalese.")

def test_audio_detokenizer():
    # Simulate a sequence of 100 thought tokens
    batch_size = 1
    seq_len = 100
    d_model = 256
    upsample = 160
    
    latent_sequence = torch.randn(batch_size, seq_len, d_model)
    decoder = AudioDetokenizer(d_model=d_model, upsample_factor=upsample)
    
    waveform = decoder(latent_sequence)
    
    expected_length = seq_len * upsample
    assert waveform.shape == (batch_size, 1, expected_length), "La projection audio a échoué. Mauvaises dimensions de tenseur d'onde."
    assert torch.max(waveform) <= 1.0 and torch.min(waveform) >= -1.0, "Amplitude audio hors limites [-1, 1]."
    print("[PASS] Audio Detokenizer: Tenseur de forme d'onde audio (Waveform) généré avec succès.")

def test_world_3d_detokenizer():
    # Simulate a thought generating 1024 3D points
    batch_size = 1
    num_points = 1024
    d_model = 256
    
    latent_sequence = torch.randn(batch_size, num_points, d_model)
    decoder = World3DDetokenizer(d_model=d_model)
    
    point_cloud = decoder(latent_sequence)
    
    assert point_cloud.shape == (batch_size, num_points, 3), "La projection 3D a échoué. Attendu: coordonnées X,Y,Z."
    print("[PASS] World 3D Detokenizer: Nuage de points 3D (X, Y, Z) généré avec succès.")

def test_mcp_executor():
    # Simulate a single thought vector deciding to use a tool
    qpls_vector = torch.randn(1, 256)
    
    executor = MCPExecutor(d_model=256)
    json_payload = executor.generate_mcp_request(qpls_vector)
    
    # Verify we get a valid JSON string
    assert isinstance(json_payload, str)
    
    # Parse the JSON and verify the MCP JSON-RPC 2.0 structure
    parsed = json.loads(json_payload)
    assert parsed["jsonrpc"] == "2.0", "Format MCP invalide (jsonrpc absent)."
    assert "method" in parsed, "Méthode d'outil absente."
    assert "params" in parsed, "Paramètres d'outil absents."
    print(f"[PASS] MCP Executor: Payload JSON-RPC valide généré pour l'outil '{parsed['method']}'.")

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
