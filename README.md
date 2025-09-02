# Optimizing Hidden Markov Language Models

Minimal implementation of HMM language models with neural reparameterization and latent variable distillation (LVD), based on:

**"Optimizing Hidden Markov Language Models: An Empirical Study of Reparameterization and Initialization Techniques"**  
Ivan Lee and Taylor Berg-Kirkpatrick, UC San Diego  
Paper: https://aclanthology.org/2025.findings-naacl.429/

## Installation

```bash
# Clone the repository
git clone https://github.com/ivnle/hmm-optimizer
cd hmm-optimizer

# Run training directly with uv (dependencies auto-installed)
uv run python train.py --config configs/baseline.yaml
```

Dependencies are automatically installed on first run with `uv`.

## Quick Start

The repository demonstrates a 2x2 matrix of capabilities using the [tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) dataset:

|                 | No LVD      | With LVD    |
|-----------------|-------------|-------------|
| **No Reparam** | baseline.yaml | lvd.yaml     |
| **With Reparam** | neural.yaml   | neural_lvd.yaml |

### Try all configurations:

```bash
# Baseline HMM
uv run python train.py --config configs/baseline.yaml

# Neural reparameterization (MLP)
uv run python train.py --config configs/neural.yaml

# LVD initialization + training
uv run python train.py --config configs/lvd.yaml

# Combined approach
uv run python train.py --config configs/neural_lvd.yaml
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lee-berg-kirkpatrick-2025-optimizing,
    title = "Optimizing Hidden {M}arkov Language Models: An Empirical Study of Reparameterization and Initialization Techniques",
    author = "Lee, Ivan  and
      Berg-Kirkpatrick, Taylor",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.429/",
    doi = "10.18653/v1/2025.findings-naacl.429",
    pages = "7712--7723",
    ISBN = "979-8-89176-195-7",
    abstract = "Hidden Markov models (HMMs) are valuable for their ability to provide exact and tractable inference. However, learning an HMM in an unsupervised manner involves a non-convex optimization problem that is plagued by poor local optima. Recent work on scaling-up HMMs to perform competitively as language models has indicated that this challenge only increases with larger hidden state sizes. Several techniques to address this problem have been proposed, but have not be evaluated comprehensively. This study provides a comprehensive empirical analysis of two recent strategies that use neural networks to enhance HMM optimization: neural reparameterization and neural initialization. We find that (1) these techniques work effectively for scaled HMM language modeling, (2) linear reparameterizations can be as effective as non-linear ones, and (3) the strategies are complementary."
}
```