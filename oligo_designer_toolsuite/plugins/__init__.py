"""
Plugin discovery utilities for Oligo Designer Toolsuite.

This module centralizes all entry-point based plugin discovery so that
other modules (e.g. oligo_specificity_filter, oligo_property_filter)
can remain focused on their core logic.

Currently supported plugin groups:

- oligo_designer_toolsuite.specificity_filters

The general contract for specificity filter plugins is:

- The entry-point group is "oligo_designer_toolsuite.specificity_filters".
- Each entry point MUST load to a subclass of
  `oligo_designer_toolsuite.oligo_specificity_filter.SpecificityFilterBase`.
- The entry point's name is used as the identifier in registries and
  configuration files.
"""

from .plugins import discover_specificity_filter_plugins, list_specificity_filter_plugins

__all__ = ["discover_specificity_filter_plugins", "list_specificity_filter_plugins"]
