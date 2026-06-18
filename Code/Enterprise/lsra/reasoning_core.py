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
            if symbolic_gate is not None:
                is_legal = symbolic_gate.verify_latent_step(entities[0, 0], operators[0, 0])
                if not is_legal:
                    trajectories.append(next_state)
                    return torch.stack(trajectories), False # Return False to signal backtrack needed
            
            # 3. Oscillation/Stagnation Detection (Gödel's Gate Crash Prevention)
            # If the state repeats or stagnates but confidence is still low, we are in a paradox loop.
            if t >= 1:
                prev_state = trajectories[-1]
                # Compute cosine similarity between current state and previous state
                cos_sim = torch.nn.functional.cosine_similarity(next_state, prev_state, dim=-1)
                # If highly similar (stagnating) and confidence is low, break early
                # Average the cosine similarity across the batch
                avg_cos_sim = cos_sim.mean().item()
                confidence = torch.sigmoid(metadata[0, 0, 0]).item() # taking first item of batch for proto
                if avg_cos_sim > 0.95 and confidence < self.grok_threshold:
                    print("[WARNING] Stagnation Épistémique détectée (Paradoxe). Arrêt d'urgence du LSRA.")
                    trajectories.append(next_state)
                    return torch.stack(trajectories), False # Fail the thought process

            trajectories.append(next_state)
            current_state = next_state
            
            # 4. Check Convergence
            confidence = torch.sigmoid(metadata[0, 0, 0]).item()
            if confidence >= self.grok_threshold:
                break
                
        return torch.stack(trajectories), True
