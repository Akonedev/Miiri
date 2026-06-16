import torch
from torch.utils.data import IterableDataset
import json

class NeuroSymbolicPrimitiveDataset(IterableDataset):
    """
    Dataset optimisé pour l'ingestion de primitives neuro-symboliques.
    Conçu selon les recommandations expertes : Iterable, Orthogonal Sparsity, Raw Tensors.
    """
    def __init__(self, file_path, d_model=256):
        super(NeuroSymbolicPrimitiveDataset).__init__()
        self.file_path = file_path
        self.d_model = d_model
        assert self.d_model == 256, "QPLS architecture requires d_model=256"

    def _process_line(self, line):
        data = json.loads(line)
        primitive_type = data['type']
        
        # In a real system, the value would be a 64-d semantic hash.
        # Here we pad the mock 5-d value to 64-d for the prototype.
        raw_val = torch.tensor(data['value'], dtype=torch.float32)
        padded_val = torch.zeros(64)
        padded_val[:len(raw_val)] = raw_val
        
        # Enforce Orthogonal Sparsity (Zero-padding non-relevant segments)
        qpls_vector = torch.zeros(self.d_model)
        
        if primitive_type == "ENTITY":
            qpls_vector[0:64] = padded_val
        elif primitive_type == "PROPERTY":
            qpls_vector[64:128] = padded_val
        elif primitive_type == "OPERATOR":
            qpls_vector[128:192] = padded_val
        else:
            raise ValueError(f"Unknown primitive type: {primitive_type}")
            
        # Optional: Add metadata (e.g., source modality = TEXT)
        qpls_vector[192] = 1.0 # Source: Text
            
        return qpls_vector

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                yield self._process_line(line)

def qpls_collate_fn(batch):
    """
    Symbolic Guardrail Collate Function.
    Asserts partition purity before the data reaches the GPU.
    """
    stacked = torch.stack(batch)
    
    # Assert that at least one partition is purely zero (Sparsity check)
    # This ensures we aren't feeding entangled noise during primitive grokking.
    for vec in stacked:
        e, p, o, m = vec[0:64], vec[64:128], vec[128:192], vec[192:256]
        # For primitive grokking, exactly one semantic block should be active
        active_blocks = (e.any(), p.any(), o.any())
        assert sum(active_blocks) == 1, "Orthogonal Sparsity Violation: Primitive vector is entangled across partitions!"
        
    return stacked
