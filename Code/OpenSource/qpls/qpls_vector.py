import torch
import torch.nn as nn

class QPLSVector(nn.Module):
    """
    Quad-Partitioned Latent Semantic Space (QPLS) Vector
    Dimension d=256, strictly partitioned into 4x64 segments.
    """
    def __init__(self, d_model=256):
        super(QPLSVector, self).__init__()
        assert d_model == 256, "QPLS requires exactly d=256 for standard partition."
        self.d_model = d_model
        
    def forward(self, entities, properties, operators, metadata):
        """
        Constructs the QPLS vector from its 4 components.
        Each component must be a tensor of shape (..., 64).
        """
        assert entities.shape[-1] == 64
        assert properties.shape[-1] == 64
        assert operators.shape[-1] == 64
        assert metadata.shape[-1] == 64
        
        # Concatenate along the last dimension
        qpls = torch.cat([entities, properties, operators, metadata], dim=-1)
        return qpls

    @staticmethod
    def extract_segments(qpls_tensor):
        """
        Extracts the 4 segments from a QPLS tensor of shape (..., 256)
        """
        assert qpls_tensor.shape[-1] == 256
        entities = qpls_tensor[..., 0:64]
        properties = qpls_tensor[..., 64:128]
        operators = qpls_tensor[..., 128:192]
        metadata = qpls_tensor[..., 192:256]
        return entities, properties, operators, metadata

class InfoNCELoss(nn.Module):
    """
    Amodal Alignment Loss using InfoNCE.
    Forces representations from different modalities (Audio, Text, Vision)
    to map to the same QPLS subspace for a given concept.
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_modality_1, v_modality_2):
        """
        v_modality_1: Tensor of shape (batch_size, d)
        v_modality_2: Tensor of shape (batch_size, d)
        """
        # Normalize vectors
        v1 = nn.functional.normalize(v_modality_1, dim=-1)
        v2 = nn.functional.normalize(v_modality_2, dim=-1)
        
        # Cosine similarity matrix
        logits = torch.matmul(v1, v2.T) / self.temperature
        
        # Labels are the diagonal (batch indices)
        batch_size = v1.shape[0]
        labels = torch.arange(batch_size, device=v1.device)
        
        loss = self.criterion(logits, labels)
        return loss
