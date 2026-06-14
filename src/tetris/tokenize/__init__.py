"""Tokenization (D9.2): segment spec, window sampler, and the pure assemble
function (the heart of the collator). No tokenization happens at packing time —
``SegmentSpec`` carries everything needed to compute segment length ``S``."""
