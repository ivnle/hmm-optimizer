"""HMM Optimizer: Minimal implementation of HMM language models with reparameterization and LVD."""

from .hmm import HMM
from .reparameterization import (
    MLPReparameterization,
    ReparameterizedHMM
)
from .lvd import SimpleTeacher, initialize_with_lvd, LVDLoss

__all__ = [
    "HMM",
    "MLPReparameterization",
    "ReparameterizedHMM",
    "SimpleTeacher",
    "initialize_with_lvd",
    "LVDLoss"
]