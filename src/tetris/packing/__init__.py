"""Packing layer: the stateless ``pack()`` collator (S6); reservoir + scheduler
(S11). The collator's signature is frozen — both the trivial path and the
reservoir path materialize the same ``Batch`` from a caller-provided grouping."""
