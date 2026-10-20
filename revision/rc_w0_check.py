"""Scale check for the reconstructed initialisation."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
from src.models import M4_FlattenFC


def w0(seed):
    torch.manual_seed(seed)
    m = M4_FlattenFC(num_classes=11)
    return m.state_dict()["classifier.1.weight"].detach().clone()


a = w0(42)
b = w0(43)
print("unrelated inits (seed 42 vs 43): rel dist = %.4f" % ((a - b).norm() / a.norm()))
print("A wrong W(0) would therefore read ~1.4, not 0.18.")
