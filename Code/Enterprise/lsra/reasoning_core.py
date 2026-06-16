import torch
import torch.nn as nn
from Code.OpenSource.qpls.qpls_vector import QPLSVector

class LSRAReasoningCore(nn.Module):
    """
    Latent-to-Symbolic Recurrent Architecture (LSRA)
    Executes "Test-Time Compute" by unrolling a shared Transformer block.
    """
    def __init__(self, d_model=256, n_heads=8, max_iters=64, grok_threshold=0.95):
        super(LSRAReasoningCore, self).__init__()
        self.max_iters = max_iters
        self.grok_threshold = grok_threshold
        
        # Shared Transformer Block for Latent Reasoning
        self.recurrent_block = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        
        # Layer Normalization for stability across deep recurrence
        self.norm = nn.LayerNorm(d_model)

    def forward(self, qpls_sequence, symbolic_gate=None):
        """
        qpls_sequence: Tensor of shape (batch, seq_len, 256)
        """
        current_state = qpls_sequence
        trajectories = [current_state]
        
        for t in range(self.max_iters):
            # 1. Propose next state
            next_state = self.recurrent_block(current_state)
            next_state = self.norm(next_state)
            
            # Extract QPLS components
            entities, properties, operators, metadata = QPLSVector.extract_segments(next_state)
            
            # 2. Symbolic Verification Gate
            # For this PyTorch simulation, we pass if no gate is provided, 
            # otherwise we verify. If invalid, ACSP loss will penalize during training.
            if symbolic_gate is not None:
                # Assuming batch_size=1, seq_len=1 for simple prototype
                is_legal = symbolic_gate.verify_latent_step(entities[0, 0], operators[0, 0])
                if not is_legal:
                    # In inference, we would trigger a backtrack.
                    # In training, we return the trajectory to compute the penalty loss.
                    trajectories.append(next_state)
                    return torch.stack(trajectories), False # Return False to signal backtrack needed
            
            trajectories.append(next_state)
            current_state = next_state
            
            # 3. Check Convergence (Metadata dimension 0 is confidence)
            # Sigmoid used to constrain confidence between 0 and 1
            confidence = torch.sigmoid(metadata[0, 0, 0]).item()
            if confidence >= self.grok_threshold:
                break
                
        return torch.stack(trajectories), True
