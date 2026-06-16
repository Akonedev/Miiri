import torch
import pytest
import sys
import os

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.OpenSource.qpls.qpls_vector import QPLSVector, InfoNCELoss
from Code.Enterprise.lsra.reasoning_core import LSRAReasoningCore
from Code.Enterprise.supervisor.symbolic_gate import SymbolicVerificationGate
from Code.Enterprise.acsp.loss import ACSPLoss

def test_qpls_vector_shape():
    qpls = QPLSVector(d_model=256)
    
    # Mock components
    entities = torch.randn(1, 1, 64)
    properties = torch.randn(1, 1, 64)
    operators = torch.randn(1, 1, 64)
    metadata = torch.randn(1, 1, 64)
    
    vector = qpls(entities, properties, operators, metadata)
    assert vector.shape == (1, 1, 256), "QPLS vector must be exactly 256 dimensions"
    
    # Test extraction
    e, p, o, m = QPLSVector.extract_segments(vector)
    assert torch.allclose(e, entities)
    assert torch.allclose(p, properties)
    assert torch.allclose(o, operators)
    assert torch.allclose(m, metadata)

def test_infonce_loss():
    loss_fn = InfoNCELoss(temperature=0.1)
    
    # Two similar modalities (e.g. text and audio representing the same concept)
    v1 = torch.randn(4, 256)
    v2 = v1 + torch.randn(4, 256) * 0.01 # Small noise
    
    loss = loss_fn(v1, v2)
    assert loss.item() > 0, "Loss should be positive"

def test_lsra_and_symbolic_gate():
    # Setup Dictionary
    # We mock it: Operator 0 requires Entity 1.
    semantic_dict = {
        0: {"requires": 1, "yields": 2}
    }
    gate = SymbolicVerificationGate(semantic_dict)
    
    core = LSRAReasoningCore(d_model=256, n_heads=4, max_iters=3)
    loss_fn = ACSPLoss(backtrack_penalty=1000.0)
    
    # Create an illegal state:
    # Entity ID is 2, but Operator ID 0 requires Entity 1.
    qpls_illegal = torch.zeros(1, 1, 256)
    qpls_illegal[0, 0, 2] = 10.0 # Entity argmax = 2
    qpls_illegal[0, 0, 128] = 10.0 # Operator argmax = 0 (128 offset)
    
    trajectories, is_legal = core(qpls_illegal, symbolic_gate=gate)
    
    assert not is_legal, "The gate should have rejected this state."
    
    loss = loss_fn(trajectories, is_legal)
    # The loss should be at least 1000 due to the backtrack penalty
    assert loss.item() >= 1000.0, "ACSP Loss did not apply the backtrack penalty correctly."

if __name__ == "__main__":
    pytest.main(["-v", __file__])
