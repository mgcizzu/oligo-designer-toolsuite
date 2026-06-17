"""
Pipeline package for oligo design workflows.

Pipeline classes are imported lazily so users can import a single pipeline without
triggering the dependency chain of all other workflow modules.
"""

__all__ = [
    "GenomicRegionGenerator",
    "OligoSeqProbeDesigner",
    "CycleHCRProbeDesigner",
    "FlexProbeDesigner",
    "ScrinshotProbeDesigner",
    "ScrinshotISSProbeDesigner",
    "CustomSequenceScrinshotISSProbeDesigner",
    "SeqFishPlusProbeDesigner",
    "MerfishProbeDesigner",
]


def __getattr__(name):
    mapping = {
        "GenomicRegionGenerator": "._genomic_region_generator",
        "OligoSeqProbeDesigner": "._oligo_seq_probe_designer",
        "CycleHCRProbeDesigner": "._cycle_hcr_probe_designer",
        "FlexProbeDesigner": "._flex_probe_designer",
        "ScrinshotProbeDesigner": "._scrinshot_probe_designer",
        "ScrinshotISSProbeDesigner": "._scrinshot_iss_probe_designer",
        "CustomSequenceScrinshotISSProbeDesigner": "._custom_sequence_scrinshot_iss_probe_designer",
        "SeqFishPlusProbeDesigner": "._seqfish_plus_probe_designer",
        "MerfishProbeDesigner": "._merfish_probe_designer",
    }
    if name in mapping:
        import importlib

        module = importlib.import_module(mapping[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
