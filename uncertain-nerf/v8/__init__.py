"""PURI-NeRF V8 staged implementation.

The first delivery contains only the baseline/V7.5 diagnostic gate.  The SDF,
reflection, participating-medium and posterior-UQ stages are intentionally not
imported until the baseline acceptance criteria have been measured.
"""

from .diagnostic_model import DiagnosticRadianceField
from .diagnostic_trainer import DiagnosticTrainer

__all__ = ["DiagnosticRadianceField", "DiagnosticTrainer"]
