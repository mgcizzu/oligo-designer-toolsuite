from __future__ import annotations

import warnings
from functools import lru_cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from oligo_designer_toolsuite.oligo_specificity_filter import BaseSpecificityFilter

SPECIFICITY_FILTER_ENTRYPOINT_GROUP = "oligo_designer_toolsuite.specificity_filters"


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

    discovered: dict[str, type[BaseSpecificityFilter]] = {}

    for ep in entry_points(group=SPECIFICITY_FILTER_ENTRYPOINT_GROUP):
        try:
            obj = ep.load()
        except Exception as exc:
            warnings.warn(
                f"[oligo-designer-toolsuite] Failed to load specificity filter "
                f"entry point '{ep.name}' from '{ep.value}': {exc!r}",
                RuntimeWarning,
            )
            continue

        if not isinstance(obj, type) or not issubclass(obj, BaseSpecificityFilter):
            warnings.warn(
                f"[oligo-designer-toolsuite] Entry point '{ep.name}' in group "
                f"'{SPECIFICITY_FILTER_ENTRYPOINT_GROUP}' did not resolve to a "
                f"BaseSpecificityFilter subclass (got {obj!r}). Skipping.",
                RuntimeWarning,
            )
            continue

        if ep.name in discovered:
            warnings.warn(
                f"[oligo-designer-toolsuite] Duplicate specificity filter "
                f"entry point name '{ep.name}'. Overwriting previous "
                f"definition '{discovered[ep.name]!r}' with '{obj!r}'.",
                RuntimeWarning,
            )

        discovered[ep.name] = obj

    return discovered


def list_specificity_filter_plugins() -> Mapping[str, str]:
    """
    Return a simple name -> import_path mapping for discovered specificity plugins.

    This is primarily intended for debugging / introspection.

    :return: Mapping from plugin name to the underlying import path string
    :rtype: Mapping
    """
    plugins: dict[str, str] = {}
    for ep in entry_points(group=SPECIFICITY_FILTER_ENTRYPOINT_GROUP):
        plugins[ep.name] = ep.value
    return plugins
