"""Core HMM implementation with forward-backward algorithm for SGD training."""

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class HMM(nn.Module):
    """Hidden Markov Model for gradient-based optimization."""
    
    def __init__(self, n_hidden_states: int, vocab_size: int, param_std: float = 1.0):
        super().__init__()
        self.n_hidden_states = n_hidden_states
        self.vocab_size = vocab_size
        
        # Initialize parameters with configurable standard deviation
        # Smaller values (e.g., 0.01) → near-uniform probabilities after softmax
        # Larger values (e.g., 1.0) → more peaked distributions after softmax
        
        self.em = nn.Parameter(torch.randn(n_hidden_states, vocab_size) * param_std)
        self.tm = nn.Parameter(torch.randn(n_hidden_states, n_hidden_states) * param_std)
        self.p = nn.Parameter(torch.randn(n_hidden_states) * param_std)
    
    
    @staticmethod
    def forward_algo(input_ids, em, tm, p, do_log_softmax=True):
        """
        Forward algorithm: compute P(o_1, ..., o_t, S_t = i) for all t, i.
        
        Args:
            input_ids: [batch_size, seq_len] - observed sequence
            em: [n_hidden, vocab_size] - emission matrix
            tm: [n_hidden, n_hidden] - transition matrix  
            p: [n_hidden] - initial state distribution
            do_log_softmax: whether to apply log_softmax to parameters
            
        Returns:
            alpha: [batch_size, n_hidden, seq_len] - forward probabilities
        """
        if do_log_softmax:
            em = log_softmax(em, dim=-1)
            tm = log_softmax(tm, dim=-1)
            p = log_softmax(p, dim=-1)
            
        batch_size, seq_len = input_ids.size()
        n_hidden = tm.size(0)
        device = input_ids.device
        
        alpha = torch.zeros(batch_size, n_hidden, seq_len).to(device)
        
        # Initialize: alpha[:, :, 0] = p + em[:, input_ids[:, 0]]
        alpha[:, :, 0] = p + em[:, input_ids[:, 0]].T
        
        # Recursion
        for t in range(1, seq_len):
            # alpha[:, :, t] = emission + logsumexp(alpha_prev + transition)
            alpha[:, :, t] = em[:, input_ids[:, t]].T + torch.logsumexp(
                alpha[:, :, t-1].unsqueeze(-1) + tm, dim=1
            )
        
        return alpha
    
    def backward_algo(self, input_ids):
        """
        Backward algorithm: compute P(o_t+1, ..., o_T | S_t = i).
        
        Args:
            input_ids: [batch_size, seq_len] - observed sequence
            
        Returns:
            beta: [batch_size, n_hidden, seq_len] - backward probabilities
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        beta = torch.zeros(batch_size, self.n_hidden_states, seq_len).to(device)
        # beta[:, :, -1] = 0 (in log space)
        
        for t in range(seq_len - 2, -1, -1):
            beta[:, :, t] = torch.logsumexp(
                beta[:, :, t + 1].unsqueeze(1) +  # [b, 1, h]
                self.em[:, input_ids[:, t + 1]].T.unsqueeze(1) +  # [b, 1, h]
                self.tm,  # [h, h]
                dim=-1
            )
        
        return beta
    
    @staticmethod
    def compute_loss_given_alpha(alpha):
        """
        Compute negative log likelihood from forward probabilities.
        
        Args:
            alpha: [batch_size, n_hidden, seq_len] - forward probabilities
            
        Returns:
            loss: scalar - negative log likelihood per token
        """
        batch_size, _, seq_len = alpha.size()
        log_prob = torch.logsumexp(alpha[:, :, -1], dim=1)  # [batch_size]
        log_prob = log_prob.mean()  # Average over batch
        loss = -log_prob / seq_len  # Per token
        return loss
    
    @staticmethod  
    def compute_joint_prob(input_ids, hidden_state_ids, em, tm, p, do_log_softmax=True):
        """
        Compute joint probability P(O, S) for LVD loss.
        
        Args:
            input_ids: [batch_size, seq_len] - observed sequence
            hidden_state_ids: [batch_size, seq_len] - hidden state sequence
            em, tm, p: HMM parameters
            do_log_softmax: whether to apply log_softmax
            
        Returns:
            log_joint: [batch_size] - log P(O, S)
        """
        if do_log_softmax:
            em = log_softmax(em, dim=-1)
            tm = log_softmax(tm, dim=-1)
            p = log_softmax(p, dim=-1)
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        log_joint = torch.zeros(batch_size).to(device)
        
        # Initial state probability
        log_joint += p[hidden_state_ids[:, 0]]
        
        # Transition probabilities
        transitions = tm[hidden_state_ids[:, :-1], hidden_state_ids[:, 1:]]  # [b, t-1]
        log_joint += transitions.sum(dim=-1)
        
        # Emission probabilities
        emissions = em[hidden_state_ids]  # [b, t, v]
        emissions = torch.gather(emissions, 2, input_ids.unsqueeze(-1)).squeeze(-1)  # [b, t]
        log_joint += emissions.sum(dim=-1)
        
        return log_joint
    
    def forward(self, input_ids):
        """
        Forward pass for training.
        
        Args:
            input_ids: [batch_size, seq_len] - input token ids
            
        Returns:
            loss: scalar - negative log likelihood
        """
        # Compute forward algorithm with gradients
        alpha = self.forward_algo(input_ids, self.em, self.tm, self.p, do_log_softmax=True)
        loss = self.compute_loss_given_alpha(alpha)
        return loss