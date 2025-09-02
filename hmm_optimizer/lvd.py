"""Latent Variable Distillation (LVD) for HMM initialization and training."""

import torch
import torch.nn as nn
import numpy as np
import faiss
from typing import Optional, Tuple
from tqdm import tqdm


class SimpleTeacher(nn.Module):
    """
    Simple teacher model for Latent Variable Distillation (LVD).
    
    Can either load a pretrained model (e.g., TinyLlama) or use random embeddings.
    Even random embeddings help optimization!
    
    Default: TinyLlama-1.1B checkpoints are commonly used as teachers in the original paper.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int = 768, 
                 checkpoint_path: Optional[str] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim  # Keep this for backward compatibility
        self.target_hidden_dim = hidden_dim  # Desired output dimension
        self.pretrained_model = None
        self.pca_matrix = None  # Will store PCA transformation
        self.pca_mean = None    # PCA centering mean
        
        if checkpoint_path:
            try:
                # Try to load from HuggingFace
                from transformers import AutoModel
                
                print(f"Loading teacher model from {checkpoint_path}...")
                self.pretrained_model = AutoModel.from_pretrained(checkpoint_path)
                self.pretrained_model.eval()  # Set to eval mode
                
                # Get the actual hidden dimension from the model
                if hasattr(self.pretrained_model.config, 'hidden_size'):
                    self.model_hidden_dim = self.pretrained_model.config.hidden_size
                else:
                    # Fallback for different model architectures
                    self.model_hidden_dim = self.pretrained_model.config.dim
                
                print(f"  Loaded model with hidden_size={self.model_hidden_dim}")
                
                # Note: PCA projection will be computed during k-means initialization
                if self.model_hidden_dim != self.target_hidden_dim:
                    print(f"  Will use PCA projection: {self.model_hidden_dim} -> {self.target_hidden_dim}")
                
                # Freeze pretrained model parameters
                for param in self.pretrained_model.parameters():
                    param.requires_grad = False
                    
            except Exception as e:
                print(f"Warning: Could not load pretrained model from {checkpoint_path}")
                print(f"  Error: {e}")
                print(f"  Falling back to random initialization (still effective!)")
                self.pretrained_model = None
        
        # Always create simple embedding-based teacher as fallback
        if self.pretrained_model is None:
            print(f"Using random teacher with hidden_dim={hidden_dim}")
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Will be set after k-means clustering
        self.cluster_centers = None
    
    def set_pca_projection(self, pca_matrix, pca_mean):
        """
        Store PCA projection matrix and mean for dimensionality reduction.
        
        Args:
            pca_matrix: [target_dim, model_dim] - PCA projection matrix
            pca_mean: [model_dim] - mean for centering
        """
        self.pca_matrix = pca_matrix
        self.pca_mean = pca_mean
        print(f"  PCA projection matrix set: {pca_matrix.shape}")
    
    def forward(self, input_ids):
        """
        Get embeddings for input tokens.
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, seq_len, target_hidden_dim]
        """
        if self.pretrained_model is not None:
            # Use pretrained model
            with torch.no_grad():  # No gradients needed for teacher
                outputs = self.pretrained_model(input_ids=input_ids)
                # Get hidden states from last layer
                x = outputs.last_hidden_state  # [batch_size, seq_len, model_hidden_dim]
                
                # Apply PCA projection if needed
                if self.pca_matrix is not None:
                    # Center the data
                    x_centered = x - self.pca_mean.unsqueeze(0).unsqueeze(0)
                    # Apply PCA projection: [b, t, model_dim] @ [model_dim, target_dim] -> [b, t, target_dim]
                    x = torch.matmul(x_centered, self.pca_matrix.T)
        else:
            # Use simple embedding-based model
            x = self.embed(input_ids)
            x = self.layers(x)
        
        return x
    
    def get_cluster_assignment(self, embeddings):
        """
        Assign embeddings to nearest cluster centers.
        
        Args:
            embeddings: [batch_size, seq_len, hidden_dim]
            
        Returns:
            cluster_ids: [batch_size, seq_len]
        """
        if self.cluster_centers is None:
            raise ValueError("Must run k-means clustering first")
        
        batch_size, seq_len, _ = embeddings.shape
        
        # Reshape for distance computation
        hidden_dim = embeddings.shape[-1]  # Use actual dimension from embeddings
        emb_flat = embeddings.reshape(-1, hidden_dim)  # [b*t, d]
        
        # Compute distances to all cluster centers
        distances = torch.cdist(emb_flat, self.cluster_centers)  # [b*t, k]
        
        # Find nearest cluster
        cluster_ids = torch.argmin(distances, dim=-1)  # [b*t]
        cluster_ids = cluster_ids.reshape(batch_size, seq_len)  # [b, t]
        
        return cluster_ids


# Removed compute_all_cluster_assignments - now computing dynamically during training


def run_kmeans(teacher: SimpleTeacher, dataloader, n_clusters: int, 
               n_tokens: int = 10_000_000, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run k-means clustering on teacher embeddings.
    
    Args:
        teacher: Teacher model
        dataloader: Data loader for getting embeddings
        n_clusters: Number of clusters (should equal n_hidden_states)
        n_tokens: Number of tokens to use for clustering
        seed: Random seed
        
    Returns:
        cluster_centers: [n_clusters, target_hidden_dim] - cluster centroids
        vocab_embeddings: [vocab_size, target_hidden_dim] - vocabulary embeddings
    """
    device = next(teacher.parameters()).device
    embeddings_list = []
    tokens_seen = 0
    
    print(f"Collecting embeddings for k-means clustering (target: {n_tokens:,} tokens)...")
    teacher.eval()
    
    # Determine if we need to collect raw embeddings for PCA
    needs_pca = (teacher.pretrained_model is not None and 
                 hasattr(teacher, 'model_hidden_dim') and 
                 teacher.model_hidden_dim != teacher.target_hidden_dim)
    
    # Calculate total batches needed
    batch_size = next(iter(dataloader))['input_ids'].shape[0]
    seq_len = next(iter(dataloader))['input_ids'].shape[1]
    tokens_per_batch = batch_size * seq_len
    total_batches = min(len(dataloader), (n_tokens + tokens_per_batch - 1) // tokens_per_batch)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=total_batches, desc="Collecting embeddings"):
            if tokens_seen >= n_tokens:
                break
            
            input_ids = batch['input_ids'].to(device)
            
            if needs_pca:
                # Get raw embeddings before PCA
                outputs = teacher.pretrained_model(input_ids=input_ids)
                embeddings = outputs.last_hidden_state  # [batch_size, seq_len, model_hidden_dim]
                emb_flat = embeddings.cpu().numpy().reshape(-1, teacher.model_hidden_dim)
            else:
                # Get embeddings (already correct dimension)
                embeddings = teacher(input_ids)  # [batch_size, seq_len, target_hidden_dim]
                emb_flat = embeddings.cpu().numpy().reshape(-1, teacher.target_hidden_dim)
            
            embeddings_list.append(emb_flat)
            tokens_seen += emb_flat.shape[0]
    
    # Concatenate all embeddings
    X = np.concatenate(embeddings_list, axis=0)[:n_tokens]
    print(f"Collected {X.shape[0]} embeddings of dimension {X.shape[1]}")
    
    # Apply PCA if needed
    if needs_pca:
        print(f"Applying PCA: {teacher.model_hidden_dim} -> {teacher.target_hidden_dim}")
        X, pca_matrix, pca_mean = apply_pca(X, teacher.target_hidden_dim)
        # Store PCA parameters in teacher
        teacher.set_pca_projection(
            torch.tensor(pca_matrix).to(device),
            torch.tensor(pca_mean).to(device)
        )
        print(f"After PCA: {X.shape}")
    
    # Run k-means
    print(f"Running k-means with {n_clusters} clusters...")
    kmeans = faiss.Kmeans(
        d=X.shape[1],  # Use actual dimension after PCA
        k=n_clusters,
        niter=20,
        nredo=3,
        seed=seed,
        verbose=True,
        gpu=False  # Use CPU version for simplicity
    )
    kmeans.train(X.astype(np.float32))
    
    # Get cluster centers
    cluster_centers = torch.tensor(kmeans.centroids).to(device)
    
    # Get vocabulary embeddings
    if teacher.pretrained_model is not None:
        # For pretrained models, get the embedding layer
        if hasattr(teacher.pretrained_model, 'get_input_embeddings'):
            embed_layer = teacher.pretrained_model.get_input_embeddings()
            vocab_embeddings = embed_layer.weight.data.clone()  # [vocab_size, model_hidden_dim]
            
            # Apply PCA to vocab embeddings if needed
            if teacher.pca_matrix is not None:
                vocab_embeddings = vocab_embeddings - teacher.pca_mean.unsqueeze(0)
                vocab_embeddings = torch.matmul(vocab_embeddings, teacher.pca_matrix.T)
        else:
            # Fallback: create random vocab embeddings
            print("Warning: Could not get vocab embeddings from model, using random")
            vocab_embeddings = torch.randn(teacher.vocab_size, teacher.target_hidden_dim).to(device)
    else:
        # Random teacher: use the embedding layer
        vocab_embeddings = teacher.embed.weight.data.clone()
    
    print(f"Cluster centers shape: {cluster_centers.shape}")
    print(f"Vocabulary embeddings shape: {vocab_embeddings.shape}")
    
    # Store cluster centers in teacher
    teacher.cluster_centers = cluster_centers
    
    return cluster_centers, vocab_embeddings


def apply_pca(X: np.ndarray, target_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply PCA to reduce dimensionality.
    
    Args:
        X: [n_samples, original_dim] - input data
        target_dim: desired output dimension
        
    Returns:
        X_reduced: [n_samples, target_dim] - reduced data
        pca_matrix: [target_dim, original_dim] - projection matrix
        pca_mean: [original_dim] - mean for centering
    """
    # Compute mean for centering
    pca_mean = X.mean(axis=0)
    X_centered = X - pca_mean
    
    # Use faiss PCA implementation (matches original)
    pca = faiss.PCAMatrix(X.shape[1], target_dim)
    pca.train(X.astype(np.float32))
    X_reduced = pca.apply_py(X.astype(np.float32))
    
    # Extract PCA projection matrix
    pca_matrix = faiss.vector_to_array(pca.A).reshape(target_dim, X.shape[1])
    
    return X_reduced, pca_matrix, pca_mean


def initialize_with_lvd(model, teacher: SimpleTeacher, dataloader, 
                       n_tokens: int = 10_000_000, seed: int = 42):
    """
    Initialize HMM or reparameterized model using LVD.
    
    Args:
        model: HMM or reparameterized model to initialize
        teacher: Teacher model
        dataloader: Data loader
        n_tokens: Number of tokens for k-means
        seed: Random seed
    """
    # Determine number of hidden states and target dimension
    if hasattr(model, 'n_hidden_states'):
        n_hidden = model.n_hidden_states
        vocab_size = model.vocab_size
        target_hidden_dim = teacher.target_hidden_dim  # Use teacher's default
    elif hasattr(model, 'reparam'):
        n_hidden = model.reparam.n_hidden_states
        vocab_size = model.reparam.vocab_size
        # IMPORTANT: Use reparameterization's hidden_dim for compatibility
        target_hidden_dim = model.reparam.hidden_dim
        # Update teacher's target dimension to match
        if teacher.target_hidden_dim != target_hidden_dim:
            print(f"Note: Adjusting LVD dimension from {teacher.target_hidden_dim} to {target_hidden_dim} to match reparameterization")
        teacher.target_hidden_dim = target_hidden_dim
    else:
        raise ValueError("Cannot determine model dimensions")
    
    # Run k-means clustering
    cluster_centers, vocab_embeddings = run_kmeans(
        teacher, dataloader, n_hidden, n_tokens, seed
    )
    
    # Note: We do NOT initialize model parameters with cluster centers
    # The model keeps its normal initialization
    # Cluster centers are only used by the teacher for mapping embeddings to cluster IDs
    
    print(f"K-means clustering complete: found {n_hidden} clusters for LVD supervision")


class LVDLoss:
    """Compute LVD loss for training with teacher guidance."""
    
    @staticmethod
    def compute_loss_with_fixed_assignments(input_ids, cluster_ids, em, tm, p):
        """
        Compute LVD loss using pre-computed fixed cluster assignments.
        
        Args:
            input_ids: [batch_size, seq_len] - observed tokens
            cluster_ids: [batch_size, seq_len] - pre-computed cluster assignments
            em, tm, p: HMM parameters
            
        Returns:
            loss: scalar - LVD loss
        """
        from .hmm import HMM
        
        # Convert cluster_ids to correct dtype if needed
        if cluster_ids.dtype != torch.long:
            cluster_ids = cluster_ids.long()
        
        # Compute joint probability P(O, S)
        log_joint = HMM.compute_joint_prob(
            input_ids, cluster_ids, em, tm, p, do_log_softmax=True
        )
        
        # Return negative log likelihood
        loss = -log_joint.mean() / input_ids.size(1)
        return loss
    
    @staticmethod
    def compute_loss(input_ids, teacher, em, tm, p):
        """
        Compute LVD loss: -log P(O, S) where S comes from teacher clustering.
        [DEPRECATED - use compute_loss_with_fixed_assignments instead]
        
        Args:
            input_ids: [batch_size, seq_len] - observed tokens
            teacher: Teacher model with cluster assignments
            em, tm, p: HMM parameters
            
        Returns:
            loss: scalar - LVD loss
        """
        from .hmm import HMM
        
        # Get teacher embeddings and cluster assignments
        with torch.no_grad():
            embeddings = teacher(input_ids)
            cluster_ids = teacher.get_cluster_assignment(embeddings)
        
        # Compute joint probability P(O, S)
        log_joint = HMM.compute_joint_prob(
            input_ids, cluster_ids, em, tm, p, do_log_softmax=True
        )
        
        # Return negative log likelihood
        loss = -log_joint.mean() / input_ids.size(1)
        return loss