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
