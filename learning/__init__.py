"""Training objectives and adaptation algorithms."""

from .coral import fit_coral_mlp, load_coral_checkpoint
from .real_cnn import fit_real_cnn
from .supervised import fit_sklearn

__all__ = ["fit_coral_mlp", "fit_real_cnn", "fit_sklearn", "load_coral_checkpoint"]
