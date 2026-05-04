"""WavLM audio encoder wrapper (stub — implementation lands in the next round).

This module will host two extraction backends:

* ``WavLMEncoder`` — frozen ``microsoft/wavlm-base-plus`` over raw 16 kHz
  waveforms; produces (L, 768) sequences at ~50 Hz.
* ``COVAREPSequenceReader`` — passthrough that pulls per-segment COVAREP
  sequences from CMU-MOSEI's ``CMU_MOSEI_COVAREP.csd`` (74-dim, 100 Hz).

Both expose the same ``encode`` interface so the extraction script can
dispatch on dataset.
"""
