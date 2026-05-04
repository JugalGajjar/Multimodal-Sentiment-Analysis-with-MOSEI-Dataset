"""VideoMAEv2 visual encoder wrapper (stub — implementation lands in the next round).

This module will host two extraction backends:

* ``VideoMAEEncoder`` — frozen ``OpenGVLab/VideoMAEv2-Base`` over uniformly
  sampled 16-frame clips; produces (1568, 768) sequences (8 temporal × 14 ×
  14 spatial patches).
* ``OpenFace2SequenceReader`` — passthrough that pulls per-segment OpenFace2
  sequences from CMU-MOSEI's ``CMU_MOSEI_VisualOpenFace2.csd`` (35-dim,
  ~30 Hz).

Both expose the same ``encode`` interface so the extraction script can
dispatch on dataset.
"""
