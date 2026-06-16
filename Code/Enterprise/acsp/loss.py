import torch
import torch.nn as nn

class ACSPLoss(nn.Module):
    """
    Amodal Consistency & Step Penalty (ACSP) Loss
    L_CausalRigor = a * L_align + b * L_step + y * L_sparse + d * L_consist
    """
    def __init__(self, backtrack_penalty=1000.0, sparsity_lambda=0.01):
        super(ACSPLoss, self).__init__()
        self.backtrack_penalty = backtrack_penalty
        self.sparsity_lambda = sparsity_lambda

    def forward(self, trajectories, is_legal):
        """
        trajectories: Tensor of shape (num_steps, batch, seq_len, 256)
        is_legal: Boolean indicating if the final step was approved by the Symbolic Gate
        """
        # 1. Step Validity Penalty (L_step)
        # If the recurrent loop ended in an illegal state, apply massive penalty
        l_step = 0.0
        if not is_legal:
            l_step = self.backtrack_penalty
            
        # 2. Sparsity Constraint (L_sparse)
        # L1 regularization to prevent noisy dimensions
        l_sparse = self.sparsity_lambda * torch.sum(torch.abs(trajectories))
        
        # Total Causal Rigor Loss
        # (L_align and L_consist are computed separately during amodal projection phase)
        l_total = l_step + l_sparse
        return l_total
