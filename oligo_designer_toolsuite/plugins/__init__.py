"""
Plugin discovery utilities for Oligo Designer Toolsuite.

This module centralizes all entry-point based plugin discovery.

Currently supported plugin groups:

- oligo_designer_toolsuite.specificity_filters
- oligo_designer_toolsuite.hybridization_probability_models

The entry point's name is used as the identifier in registries and configuration files.

Requirements for specificity filter plugins are:

- The entry-point group is "oligo_designer_toolsuite.specificity_filters".
- Each entry point MUST load to a subclass of
  `oligo_designer_toolsuite.oligo_specificity_filter.SpecificityFilterBase`.

Requirements for hybridization probability model plugins are:

- The entry-point group is "oligo_designer_toolsuite.hybridization_probability_models".
- Each entry point MUST load to a subclass of
  `oligo_designer_toolsuite.oligo_specificity_filter.HybridizationProbabilityModel`.
"""

from .plugins import (
    discover_hybridization_models,
    discover_specificity_filter_plugins,
    list_hybridization_probability_model_plugins,
    list_specificity_filter_plugins,
)

__all__ = [
    "discover_specificity_filter_plugins",
    "discover_hybridization_models",
    "list_specificity_filter_plugins",
    "list_hybridization_probability_model_plugins",
]
