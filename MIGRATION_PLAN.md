# LVD Migration Plan

## Overview
This document outlines the correct implementation of Latent Variable Distillation (LVD) based on the paper "Scaling Up Probabilistic Circuits by Latent Variable Distillation" and the original litgpt implementation. Our current port has critical issues that need to be fixed.

## Correct LVD Workflow

### Step 1: Extract Embeddings
- **Input**: Training sequences of tokens
- **Process**: Pass each sequence through a pretrained transformer (e.g., TinyLlama, BERT)
- **Output**: For each sequence of k tokens → k contextualized embeddings
- **Example**: 
  - Sequence: `[token_1, token_2, ..., token_k]`
  - Embeddings: `[emb_1, emb_2, ..., emb_k]` where each emb_i is d-dimensional (e.g., 768 for BERT, 2048 for TinyLlama)
- **Scale**: For N sequences of length L → collect N×L total embeddings

### Step 2: K-means Clustering
- **Input**: All N×L embeddings from Step 1
- **Process**: Run k-means clustering with K clusters (K = number of HMM hidden states)
- **Output**: K cluster centers
- **Implementation**: Uses FAISS library for efficiency

### Step 3: Assign Cluster IDs
- **Input**: All training sequences and cluster centers
- **Process**: For each token's embedding, find the nearest cluster center
- **Output**: Each token gets a cluster ID ∈ {1, ..., K}
- **Key Point**: These assignments are FIXED and used as supervised latent states S throughout LVD training

### Step 4: Train with LVD Loss
- **Duration**: Fixed number of tokens (e.g., 5M tokens)
- **Objective**: Maximize `log P(O,S)` where:
  - O = observed tokens
  - S = fixed cluster assignments from Step 3
- **Loss Computation**: Direct computation using fixed S (see details below)
- **Optimization**: Standard gradient descent (AdamW)
- **IMPORTANT**: This is a CLEAN BREAK - only LVD loss, no base HMM loss

### Step 5: Switch to Standard Training
- **Trigger**: After completing N tokens of LVD training
- **Objective**: Maximize `log P(O)` using forward algorithm
- **Loss Computation**: Forward algorithm marginalizing over all possible S
- **IMPORTANT**: This is a CLEAN BREAK - only base HMM loss, no LVD loss

### Step 6: Continue Until Convergence
- Continue standard HMM training until convergence or max iterations

## Computing P(O,S) vs P(O)

### P(O,S) - Joint Probability (LVD Phase)

For sequence O = [o₁, o₂, ..., oₜ] and given states S = [s₁, s₂, ..., sₜ]:

```
P(O,S) = p(s₁) × em[s₁,o₁] × tm[s₁,s₂] × em[s₂,o₂] × ... × tm[sₜ₋₁,sₜ] × em[sₜ,oₜ]
```

In log space:
```python
log P(O,S) = log p(s₁) + log em[s₁,o₁] + Σᵢ [log tm[sᵢ,sᵢ₊₁] + log em[sᵢ₊₁,oᵢ₊₁]]
```

**Advantages**:
- Direct lookup using fixed S
- No marginalization needed
- Very fast computation
- Provides strong supervision signal

### P(O) - Marginal Probability (Standard Phase)

For sequence O = [o₁, o₂, ..., oₜ]:

```
P(O) = Σ_S P(O,S)  # Sum over ALL possible state sequences
```

Computed via Forward Algorithm:
```python
# Initialize
α₁(s) = p(s) × em[s,o₁] for all states s

# Recurse
αₜ(s) = Σₛ' [αₜ₋₁(s') × tm[s',s] × em[s,oₜ]] for t > 1

# Final
P(O) = Σₛ αₜ(s)
```

**Characteristics**:
- Must consider all possible state sequences
- Requires forward algorithm
- More computationally expensive
- Allows model to learn optimal state assignments

## Current Implementation Issues

### Issue 1: No LVD Training Phase
**Current**: `initialize_with_lvd()` only runs k-means and sets embeddings
**Problem**: Does NOT actually train with P(O,S) for N tokens
**Impact**: Missing the core benefit of LVD - supervised training phase

### Issue 2: Incorrect Loss Usage
**Current**: If `init_only=False`, uses LVD loss for ALL epochs
**Problem**: Should only use LVD loss for initial N tokens, then switch
**Impact**: Model never learns to optimize P(O) properly

### Issue 3: Missing Cluster Assignment Storage
**Current**: Cluster assignments computed but not stored for training
**Problem**: Cannot compute P(O,S) without stored assignments
**Impact**: Cannot implement proper LVD loss

## Required Code Changes

### 1. Store Cluster Assignments
- After k-means, store cluster assignments for all training data
- Create a mapping: sequence_id → cluster_assignments

### 2. Implement Two-Phase Training
- Phase 1: Train with LVD loss for `pretrain_tokens`
  - Use stored cluster assignments
  - Compute log P(O,S) directly
- Phase 2: Switch to standard HMM loss
  - Use forward algorithm
  - Compute log P(O)

### 3. Track Token Count
- Count tokens processed during training
- Switch from Phase 1 to Phase 2 at threshold
- Log the switch clearly

### 4. Fix Configuration
- Add `pretrain_tokens` parameter (default: 5M)
- Remove `init_only` flag (misleading)
- Add clear phase indicators

### 5. Update Loss Computation
- Implement `compute_joint_prob` correctly for P(O,S)
- Ensure forward algorithm is used for P(O)
- No interpolation - clean break between phases

## Validation Plan

1. Verify k-means clustering produces reasonable clusters
2. Confirm cluster assignments are stored and accessible
3. Check that loss switches at correct token count
4. Monitor loss curves - should see:
   - Fast decrease during LVD phase (supervised)
   - Potential small increase at switch
   - Continued decrease during standard phase
5. Compare final perplexity with and without LVD

## Expected Outcomes

With correct LVD implementation:
- Faster initial convergence due to supervised latent states
- Better final perplexity than vanilla HMM
- Ability to scale to larger hidden state sizes (512, 1024, 2048)
- Similar benefits for both vanilla and neural-reparameterized HMMs