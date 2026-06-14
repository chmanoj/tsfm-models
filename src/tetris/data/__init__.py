"""Data layer: loader contract, synthetic generators, stand-in loaders.

The base loader emits raw items; everything downstream (window sampler,
reservoir, collator, model) consumes only the item tuple, so swapping the real
loaders in behind ``build_loader`` changes nothing downstream (D9/D13, §5.2).
"""
