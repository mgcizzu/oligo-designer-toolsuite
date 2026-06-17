"""
Top-level package for the oligo designer toolsuite.

Subpackages are intentionally imported lazily so that lightweight workflows do not
eagerly import optional heavy dependencies at package import time.
"""

__all__ = [
    "database",
    "oligo_efficiency_filter",
    "oligo_property_filter",
    "oligo_selection",
    "oligo_specificity_filter",
    "pipelines",
    "sequence_generator",
    "utils",
]
