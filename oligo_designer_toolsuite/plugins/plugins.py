from __future__ import annotations

import warnings
from functools import lru_cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Literal, Mapping, TypeVar

if TYPE_CHECKING:
    from oligo_designer_toolsuite.oligo_specificity_filter import (
        BaseSpecificityFilter,
        HybridizationProbabilityModel,
    )

SPECIFICITY_FILTER_ENTRYPOINT_GROUP = "oligo_designer_toolsuite.specificity_filters"
HYBRIDIZATION_PROBABILITY_MODEL_ENTRYPOINT_GROUP = "oligo_designer_toolsuite.hybridization_probability_models"

T = TypeVar("T")


# mypy can't deal with abstract classes, therefore while it is defined
# here, we need to ignore the mypy errors when the function is used
# with concrete classes with ignore[type-abstract]
def _discover(
    group: str, base_class: type[T], entry_point_type: Literal["filter", "model"] = "filter"
) -> Mapping[str, type[T]]:
    discovered: dict[str, type[T]] = {}

    for ep in entry_points(group=group):
        try:
            obj = ep.load()
        except Exception as exc:
            warnings.warn(
                f"[oligo-designer-toolsuite] Failed to load {entry_point_type} "
                f"'{ep.name}' from '{ep.value}': {exc!r}",
                RuntimeWarning,
            )
            continue

        if not isinstance(obj, type) or not issubclass(obj, base_class):
            warnings.warn(
                f"[oligo-designer-toolsuite] Entry point '{ep.name}' in group "
                f"'{group}' did not resolve to a "
                f"{base_class} subclass (got {obj!r}). Skipping.",
                RuntimeWarning,
            )
            continue

        if ep.name in discovered:
            warnings.warn(
                f"[oligo-designer-toolsuite] Duplicate {entry_point_type} "
                f"entry point name '{ep.name}'. Overwriting previous "
                f"definition '{discovered[ep.name]!r}' with '{obj!r}'.",
                RuntimeWarning,
            )

        discovered[ep.name] = obj
    return discovered


@lru_cache(maxsize=1)
def discover_specificity_filter_plugins() -> Mapping[str, type[BaseSpecificityFilter]]:
    """
    Discover all externally provided specificity filters via entry points. Only entry points
    in the group "oligo_designer_toolsuite.specificity_filters" are considered.
    Entry points that fail to load, or that do not resolve to a proper BaseSpecificityFilter subclass,
    are skipped with a warning.

    :return:  Mapping from entry point name -> BaseSpecificityFilter subclass.
    :rtype: Mapping
    """
    # Local import to avoid circular import at module import time
    from oligo_designer_toolsuite.oligo_specificity_filter import BaseSpecificityFilter

    discovered = _discover(group=SPECIFICITY_FILTER_ENTRYPOINT_GROUP, base_class=BaseSpecificityFilter, entry_point_type="filter")  # type: ignore[type-abstract]

    return discovered


def _list_plugins(group: str) -> Mapping[str, str]:
    """
    Return a simple name -> import_path mapping for discovered plugins.

    This is primarily intended for debugging / introspection.

    :return: Mapping from plugin name to the underlying import path string
    :rtype: Mapping
    """
    plugins: dict[str, str] = {}
    for ep in entry_points(group=group):
        plugins[ep.name] = ep.value
    return plugins


def list_specificity_filter_plugins() -> Mapping[str, str]:
    """
    Return a simple name -> import_path mapping for discovered specificity filter plugins.

    This is primarily intended for debugging / introspection.

    :return: Mapping from plugin name to the underlying import path string
    :rtype: Mapping
    """
    return _list_plugins(SPECIFICITY_FILTER_ENTRYPOINT_GROUP)


def list_hybridization_probability_model_plugins() -> Mapping[str, str]:
    """
    Return a simple name -> import_path mapping for discovered hybridization probability model plugins.

    This is primarily intended for debugging / introspection.

    :return: Mapping from plugin name to the underlying import path string
    :rtype: Mapping
    """
    return _list_plugins(HYBRIDIZATION_PROBABILITY_MODEL_ENTRYPOINT_GROUP)


@lru_cache(maxsize=1)
def discover_hybridization_models() -> Mapping[str, type[HybridizationProbabilityModel]]:
    """
    Discover all externally provided HybridizationProbabilityFilter models via entry points. Only entry points
    in the group "oligo_designer_toolsuite.specificity_filters" are considered.
    Entry points that fail to load, or that do not resolve to a proper HybridizationProbabilityModel class,
    are skipped with a warning.

    :return:  Mapping from entry point name -> HybridizationProbabilityModel class.
    :rtype: Mapping
    """
    # Local import to avoid circular import at module import time
    from oligo_designer_toolsuite.oligo_specificity_filter import HybridizationProbabilityModel

    discovered = _discover(group=HYBRIDIZATION_PROBABILITY_MODEL_ENTRYPOINT_GROUP, base_class=HybridizationProbabilityModel, entry_point_type="model")  # type: ignore[type-abstract]

    return discovered
