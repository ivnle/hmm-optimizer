"""Reparameterization module for HMM using Chiu & Rush 2020 MLP architecture."""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """MLP block with residual connection and LayerNorm (Chiu & Rush style)."""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.w = nn.Linear(n_embd, n_embd)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = self.w(x)
        x = self.relu(x) + x  # Residual connection
        x = self.layernorm(x)
        return x


class ChiuRushMLP(nn.Module):
    """MLP architecture from Chiu & Rush 2020 with residuals and LayerNorm."""
    
    def __init__(self, n_embd: int, n_layers: int = 2):
        super().__init__()
        assert n_layers > 1, "Need at least 2 layers for Chiu & Rush MLP"
        
        self.w1 = nn.Linear(n_embd, n_embd)
        self.relu1 = nn.ReLU()
        self.layers = nn.ModuleList([MLPBlock(n_embd) for _ in range(n_layers - 1)])
    
    def forward(self, x):
        x = self.w1(x)
        x = self.relu1(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MLPReparameterization(nn.Module):
    """
    MLP-based reparameterization of HMM parameters using Chiu & Rush 2020 architecture.
    
    Uses separate MLPs for emission and transition matrices (psi variant).
    """
    
    def __init__(self, n_hidden_states: int, vocab_size: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.n_hidden_states = n_hidden_states
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embeddings (using PyTorch's default initialization ~N(0,1))
        # Just one embedding matrix for hidden states and one for vocabulary
        self.hidden_state_embed = nn.Embedding(n_hidden_states, hidden_dim)
        self.vocab_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # MLPs for transforming embeddings (Chiu & Rush architecture)
        # These operate on the same hidden state embeddings in different roles
        self.mlp_emit = ChiuRushMLP(hidden_dim, n_layers)
        self.mlp_in = ChiuRushMLP(hidden_dim, n_layers)
        self.mlp_out = ChiuRushMLP(hidden_dim, n_layers)
        
        # Initial distribution projection
        self.p_proj = nn.Linear(hidden_dim, 1)
        
        # Pre-compute indices
        self.register_buffer('hidden_indices', torch.arange(n_hidden_states))
        self.register_buffer('vocab_indices', torch.arange(vocab_size))
    
    def forward(self):
        """
        Generate HMM parameters using neural networks.
        
        Returns:
            (em, tm, p): emission matrix, transition matrix, initial distribution
        """
        # Get embeddings for all states and vocabulary
        h = self.hidden_state_embed(self.hidden_indices)  # [n_hidden, hidden_dim]
        v = self.vocab_embed(self.vocab_indices)  # [vocab_size, hidden_dim]
        
        # Transform through MLPs (each MLP operates on the same hidden embeddings)
        h_emit = self.mlp_emit(h)  # [n_hidden, hidden_dim]
        h_in = self.mlp_in(h)  # [n_hidden, hidden_dim]
        h_out = self.mlp_out(h)  # [n_hidden, hidden_dim]
        
        # Compute matrices
        em = h_emit @ v.T  # [n_hidden, vocab_size]
        tm = h_in @ h_out.T  # [n_hidden, n_hidden]
        p = self.p_proj(h_in).squeeze(-1)  # [n_hidden]
        
        return em, tm, p


class ReparameterizedHMM(nn.Module):
    """Wrapper that combines reparameterization with HMM forward algorithm."""
    
    def __init__(self, reparam_model):
        super().__init__()
        self.reparam = reparam_model
    
    def forward(self, input_ids):
        """
        Forward pass for training.
        
        Args:
            input_ids: [batch_size, seq_len] - input token ids
            
        Returns:
            loss: scalar - negative log likelihood
        """
        # Generate HMM parameters
        em, tm, p = self.reparam()
        
        # Compute forward algorithm
        from .hmm import HMM
        alpha = HMM.forward_algo(input_ids, em, tm, p, do_log_softmax=True)
        loss = HMM.compute_loss_given_alpha(alpha)
        
        return loss