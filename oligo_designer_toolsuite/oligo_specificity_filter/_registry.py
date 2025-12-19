from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from oligo_designer_toolsuite.plugins import (
    discover_specificity_filter_plugins,
    list_specificity_filter_plugins,
)

from ._filter_base import BaseSpecificityFilter
from ._filter_blastn import BlastNFilter, BlastNSeedregionFilter, BlastNSeedregionSiteFilter
from ._filter_bowtie import Bowtie2Filter, BowtieFilter
from ._filter_cross_hybridization import CrossHybridizationFilter
from ._filter_exact_matches import ExactMatchFilter
from ._filter_hybridization_probability import HybridizationProbabilityFilter
from ._filter_variants import VariantsFilter


def _get_builtin_specificity_filters() -> Mapping[str, type[BaseSpecificityFilter]]:
    """
    Return a mapping of built-in specificity filter names to their classes.

    The keys are the canonical names that are referenced in config files
    or Python code. The values are the corresponding filter classes.

    :return: Mapping from filter names to their classes
    :rtype: Mapping
    """

    return {
        "exact_match": ExactMatchFilter,
        "blastn": BlastNFilter,
        "blastn_seedregion": BlastNSeedregionFilter,
        "blastn_seedregion_ligationsite": BlastNSeedregionSiteFilter,
        "bowtie": BowtieFilter,
        "bowtie2": Bowtie2Filter,
        "cross_hybridization": CrossHybridizationFilter,
        "variants": VariantsFilter,
        "hybridization_probability": HybridizationProbabilityFilter,
    }


@lru_cache(maxsize=1)
def get_specificity_filter_registry() -> Mapping[str, type[BaseSpecificityFilter]]:
    """
    Return a registry of all available specificity filters.

    This includes:
    - All built-in filters in this module, registered under short names
      like "blastn", "bowtie2", "cross_hybridization", etc.
    - All externally provided plugins discovered via entry points in the
      "oligo_designer_toolsuite.specificity_filters" group.

    :return: Mapping from filter name to filter class
    :rtype: Mapping
    """
    registry: dict[str, type[BaseSpecificityFilter]] = dict(_get_builtin_specificity_filters())

    # Overlay plugin-provided filters; they can override built-ins if they
    # intentionally reuse the same name.
    registry.update(discover_specificity_filter_plugins())

    return registry


def list_specificity_filters() -> Mapping[str, str]:
    """
    Return a simple mapping of specificity filter names to their origin.

    This is primarily a convenience / debugging helper to quickly see which
    filters are available and whether they are built-in or provided by plugins.

    :return:  Mapping from filter name to an origin string ("built-in" or "plugin:<import-path>")
    :rtype: Mapping
    """
    names: dict[str, str] = {}
    builtins = _get_builtin_specificity_filters()
    for name in builtins:
        names[name] = "built-in"

    # For plugins, we can resolve the import path via the plugin listing.
    for name, import_path in list_specificity_filter_plugins().items():
        names[name] = f"plugin:{import_path}"

    return names


def create_specificity_filter(
    name: str,
    *,
    filter_name: str | None = None,
    **kwargs: Any,
) -> BaseSpecificityFilter:
    """
    Convenience factory to construct a specificity filter by its registry name.

    :param name:  Name of the filter in the specificity filter registry,
        e.g. "blastn", "bowtie2", "cross_hybridization",
        or a plugin-defined name like "hybridization_probability".
    :type name: str
    :param filter_name  Optional `filter_name` argument passed through to the filter's
        constructor. If not given, the registry name is used.
    :type filter_name: str | None
    :param kwargs: Additional keyword arguments to the filter's constructor
    :type kwargs: dict
    :return: An instance of the requested filter.
    :rtype: BaseSpecificityFilter
    """
    registry = get_specificity_filter_registry()

    try:
        cls: type[BaseSpecificityFilter] = registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown specificity filter '{name}'. " f"Available filters: {available}") from exc

    if filter_name is None:
        filter_name = name

    return cls(filter_name=filter_name, **kwargs)
