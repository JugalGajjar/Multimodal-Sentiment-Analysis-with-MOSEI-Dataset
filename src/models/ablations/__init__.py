"""X-MoFE component-removal ablations (spec §19.1).

Mapping from spec ablation row → implementation:

    Architecture ablations (model classes here)
    -------------------------------------------
    w/o reliability gate              → XMoFENoReliability   (this package)
    w/o interaction block             → XMoFENoInteraction   (this package)
    w/o tri-modal interaction         → XMoFENoTrimodal      (this package)

    Loss ablations (XMoFE arch + a stripped loss config)
    ----------------------------------------------------
    w/o faithfulness loss             → configs/training/loss_no_faithfulness.yaml
    w/o stability loss                → configs/training/loss_no_stability.yaml
    w/o entropy regularization        → configs/training/loss_no_entropy.yaml
    w/o reliability supervision       → configs/training/loss_no_reliability.yaml

The architectural ablations are thin XMoFE subclasses that flip a single
flag and inherit everything else, so the canonical XMoFE forward stays the
primary readable code path. The loss ablations don't need separate model
classes — they're config flips.
"""

from src.models.ablations.xmofe_no_interaction import XMoFENoInteraction
from src.models.ablations.xmofe_no_reliability import XMoFENoReliability
from src.models.ablations.xmofe_no_trimodal import XMoFENoTrimodal

__all__ = [
    "XMoFENoInteraction",
    "XMoFENoReliability",
    "XMoFENoTrimodal",
]
