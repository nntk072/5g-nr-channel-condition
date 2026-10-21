"""Training objectives and adaptation algorithms."""

from .coral import fit_coral_mlp, load_coral_checkpoint
from .supervised import fit_sklearn

__all__ = ["fit_coral_mlp", "fit_sklearn", "load_coral_checkpoint"]
