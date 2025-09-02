#!/usr/bin/env python3
"""
Minimal training script for HMM language models.

Demonstrates the 2x2 matrix of capabilities:
- Baseline HMM
- Neural reparameterization
- LVD initialization
- Combined approaches
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path

from hmm_optimizer import (
    HMM,
    MLPReparameterization,
    ReparameterizedHMM,
    SimpleTeacher,
    initialize_with_lvd,
    LVDLoss
)


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class MovingAverage:
    """Track moving average of values over a window."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = []
    
    def update(self, value):
        """Add a new value and maintain window size."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get(self):
        """Get the current moving average."""
        return sum(self.values) / len(self.values) if self.values else 0
    
    def get_ppl(self):
        """Get perplexity from average loss."""
        avg = self.get()
        return np.exp(avg) if avg > 0 else 0


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config):
    """
    Load and prepare tiny_shakespeare dataset (all data for training).
    
    Returns:
        train_loader, vocab_size
    """
    from transformers import AutoTokenizer
    
    # Load tokenizer from config
    tokenizer_name = config['model'].get('tokenizer', 'meta-llama/Llama-2-7b-hf')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    
    print(f"Using {tokenizer_name} tokenizer with vocabulary size: {vocab_size}")
    
    # Load dataset
    dataset = load_dataset("karpathy/tiny_shakespeare")
    
    # Combine train and test text - use all data for training
    all_text = dataset['train']['text'][0] + ' ' + dataset['test']['text'][0]
    
    # Tokenize using Llama-2 tokenizer
    all_ids = torch.tensor(tokenizer.encode(all_text), dtype=torch.long)
    
    # Create sequences
    seq_len = config['training']['max_seq_length']
    batch_size = config['training']['batch_size']
    
    def create_sequences(ids, seq_len):
        """Split text into fixed-length sequences."""
        sequences = []
        for i in range(0, len(ids) - seq_len, seq_len):
            sequences.append(ids[i:i + seq_len])
        return torch.stack(sequences)
    
    all_sequences = create_sequences(all_ids, seq_len)
    
    print(f"Total training sequences: {all_sequences.shape}")
    
    # Calculate tokens per epoch for epoch-based parameters
    tokens_per_epoch = len(all_sequences) * seq_len
    print(f"Tokens per epoch: {tokens_per_epoch:,}")
    
    # Create simple dataset class
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return {'input_ids': self.sequences[idx]}
    
    train_dataset = SequenceDataset(all_sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, vocab_size, tokens_per_epoch


def create_model(config, vocab_size):
    """Create model based on configuration."""
    model_config = config['model']
    n_hidden = model_config['n_hidden_states']
    
    # Check for reparameterization
    if 'reparameterization' in config:
        reparam_config = config['reparameterization']
        hidden_dim = reparam_config.get('hidden_dim', 256)
        n_layers = reparam_config.get('n_layers', 2)  # Default to 2 layers like Chiu & Rush
        
        reparam = MLPReparameterization(n_hidden, vocab_size, hidden_dim, n_layers)
        model = ReparameterizedHMM(reparam)
    else:
        # Vanilla HMM
        param_std = config.get('training', {}).get('param_std', 1.0)
        model = HMM(n_hidden, vocab_size, param_std=param_std)
    
    return model


def train_epoch(model, train_loader, optimizer, config, 
                teacher=None, token_count=0, lvd_phase=False, 
                lvd_tokens=0, window_size=100):
    """
    Train for one epoch with moving average tracking and LVD phase support.
    
    Args:
        model: HMM model
        train_loader: DataLoader
        optimizer: Optimizer
        config: Config dict
        teacher: Teacher model for LVD (only needed during LVD phase)
        token_count: Current token count
        lvd_phase: Whether in LVD phase
        lvd_tokens: Total tokens for LVD phase
        window_size: Window size for moving average
        
    Returns:
        avg_loss, avg_ppl, updated_token_count, still_in_lvd_phase
    """
    model.train()
    moving_avg = MovingAverage(window_size=window_size)
    seq_len = config['training']['max_seq_length']
    batch_size = config['training']['batch_size']
    tokens_per_batch = batch_size * seq_len
    initial_lvd_phase = lvd_phase
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
        
        # Check if we should switch phases
        if lvd_phase and token_count >= lvd_tokens:
            lvd_phase = False
            print(f"\n{'='*50}")
            print(f"Completed LVD phase at {token_count:,} tokens")
            print(f"Switching to standard HMM training (optimizing P(O))")
            print(f"{'='*50}\n")
        
        # Compute loss based on current phase
        if lvd_phase:
            # LVD phase: compute cluster assignments on-the-fly
            if teacher is None:
                raise ValueError("Teacher model required for LVD phase")
            
            # Compute cluster assignments dynamically
            with torch.no_grad():
                embeddings = teacher(input_ids)
                cluster_ids = teacher.get_cluster_assignment(embeddings)
            
            if torch.cuda.is_available():
                cluster_ids = cluster_ids.cuda()
            
            # Get HMM parameters
            if hasattr(model, 'reparam'):
                em, tm, p = model.reparam()
            else:
                em, tm, p = model.em, model.tm, model.p
            
            # Compute LVD loss with fixed assignments
            loss = LVDLoss.compute_loss_with_fixed_assignments(
                input_ids, cluster_ids, em, tm, p
            )
            
            phase_label = f"LVD ({token_count:,}/{lvd_tokens:,})"
        else:
            # Standard phase: use forward algorithm
            loss = model(input_ids)
            phase_label = "Standard"
        
        # SGD/Adam update
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion with large vocab
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update token count
        token_count += tokens_per_batch
        
        # Update moving average
        moving_avg.update(loss.item())
        
        # Update progress bar with current and moving average
        progress_bar.set_postfix({
            'phase': phase_label,
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{moving_avg.get():.4f}', 
            'avg_ppl': f'{moving_avg.get_ppl():.2f}'
        })
    
    # Log phase transition if it happened this epoch
    if initial_lvd_phase and not lvd_phase:
        print(f"Phase transition occurred during this epoch")
    
    return moving_avg.get(), moving_avg.get_ppl(), token_count, lvd_phase


def main():
    parser = argparse.ArgumentParser(description='Train HMM language models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Prepare data
    train_loader, vocab_size, tokens_per_epoch = prepare_data(config)
    
    # Create model
    model = create_model(config, vocab_size)
    
    # Display model type and parameter count
    if hasattr(model, 'reparam'):
        print("Model type: Reparameterized HMM (MLP)")
    else:
        print("Model type: Vanilla HMM")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    
    # Initialize with LVD if specified
    teacher = None
    lvd_enabled = False
    lvd_tokens = 0
    
    if 'lvd' in config:
        lvd_config = config['lvd']
        lvd_enabled = True
        
        # Handle LVD tokens vs epochs
        if 'lvd_tokens' in lvd_config and 'lvd_epochs' in lvd_config:
            print("Warning: Both lvd_tokens and lvd_epochs specified. Using lvd_tokens.")
        
        if 'lvd_tokens' in lvd_config:
            lvd_tokens = lvd_config['lvd_tokens']
        elif 'lvd_epochs' in lvd_config:
            lvd_tokens = lvd_config['lvd_epochs'] * tokens_per_epoch
            print(f"Converting {lvd_config['lvd_epochs']} epochs to {lvd_tokens:,} tokens")
        else:
            lvd_tokens = 5_000_000  # default
        
        print("=" * 50)
        print("Initializing with LVD")
        print(f"  LVD tokens: {lvd_tokens:,}")
        print("=" * 50)
        
        # Create teacher model
        teacher = SimpleTeacher(
            vocab_size,
            hidden_dim=lvd_config.get('hidden_dim', 768),
            checkpoint_path=lvd_config.get('teacher_checkpoint', None)
        )
        if torch.cuda.is_available():
            teacher = teacher.cuda()
        
        # Handle k-means tokens vs epochs
        if 'kmeans_tokens' in lvd_config and 'kmeans_epochs' in lvd_config:
            print("Warning: Both kmeans_tokens and kmeans_epochs specified. Using kmeans_tokens.")
        
        if 'kmeans_tokens' in lvd_config:
            kmeans_tokens = lvd_config['kmeans_tokens']
        elif 'kmeans_epochs' in lvd_config:
            kmeans_tokens = lvd_config['kmeans_epochs'] * tokens_per_epoch
            print(f"Converting {lvd_config['kmeans_epochs']} epochs to {kmeans_tokens:,} tokens for k-means")
        else:
            kmeans_tokens = 10_000_000  # default
        
        # Initialize model with LVD
        print(f"Running k-means with {kmeans_tokens:,} tokens...")
        initialize_with_lvd(
            model, teacher, train_loader,
            n_tokens=kmeans_tokens,
            seed=args.seed
        )
        
        print(f"LVD initialization complete. Will train with P(O,S) for {lvd_tokens:,} tokens.")
        print(f"Teacher model will compute cluster assignments dynamically during training.")
    
    # Create AdamW optimizer (matching original litgpt implementation)
    lr = config['training'].get('learning_rate', 0.0004)  # Use litgpt default
    weight_decay = config['training'].get('weight_decay', 0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)  # Match litgpt defaults
    )
    print(f"Using AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
    
    # Training loop
    n_epochs = config['training'].get('n_epochs', 10)
    window_size = config['training'].get('moving_avg_window', 100)
    
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Using moving average with window size: {window_size}")
    print("=" * 50)
    
    best_loss = float('inf')
    token_count = 0
    lvd_phase = lvd_enabled  # Start in LVD phase if enabled
    
    for epoch in range(n_epochs):
        # Determine current phase for logging
        if lvd_phase:
            phase_desc = f"LVD Phase (target: {lvd_tokens:,} tokens)"
        else:
            phase_desc = "Standard Phase"
        
        print(f"\nEpoch {epoch+1}/{n_epochs} - {phase_desc}")
        
        # Train and get moving average metrics
        avg_loss, avg_ppl, token_count, lvd_phase = train_epoch(
            model, train_loader, optimizer, config,
            teacher=teacher,
            token_count=token_count,
            lvd_phase=lvd_phase,
            lvd_tokens=lvd_tokens,
            window_size=window_size
        )
        
        print(f"\nEpoch {epoch+1}/{n_epochs} Summary")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average PPL: {avg_ppl:.2f}")
        print(f"  Total tokens: {token_count:,}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ✓ New best average loss!")
        
        # Free teacher memory after LVD phase completes
        if not lvd_phase and teacher is not None:
            print("  Freeing teacher model from GPU memory...")
            del teacher
            teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("-" * 50)
    
    print("\nTraining complete!")
    print(f"Best average loss: {best_loss:.4f}")
    print(f"Best average perplexity: {np.exp(best_loss):.2f}")


if __name__ == "__main__":
    main()