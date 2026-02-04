"""
This module serves as an initializer for the package, aggregating various submodules related to genomic data processing and oligo sequence management.

Submodules:
- database: Contains utilities for handling and processing oligo and reference databases.
- oligo_efficiency_filter: Provides functions for filtering oligos based on efficiency criteria.
- oligo_property_filter: Provides functions for filtering oligos based on sequence properties.
- oligo_selection: Implements algorithms and methods for selecting oligosets.
- oligo_specificity_filter: Provides functions for filtering oligos based on specificity criteria.
- pipelines: Defines workflows and pipelines for oligo sequence design.
- sequence_generator: Provides functionality for generating genomic sequences and oligos.
- utils: Includes various utility functions and helper methods used across the package.
"""

import importlib
import types

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


def __getattr__(name: str) -> types.ModuleType:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
