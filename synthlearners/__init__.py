"""Public package interface for synthlearners.

Lean installs expose the adelie-based ``PenguinSynth`` API. Full installs also
expose the traditional ``Synth`` estimator, dynamic balancing tools, and
related result containers.
"""

from .lbw import PenguinSynth, PenguinResults

# Always available (lean installation)
__all__ = [
    "PenguinSynth", 
    "PenguinResults",
]

# Optional full installation components
try:
    from .synth import Synth, SynthResults
    __all__.extend(["Synth", "SynthResults"])
except ImportError:
    pass

try:
    from .dynbalance import DynamicBalance, DynamicBalanceResults

    __all__.extend(["DynamicBalance", "DynamicBalanceResults"])
except ImportError:
    pass
