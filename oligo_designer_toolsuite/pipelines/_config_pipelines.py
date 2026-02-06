from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from ._config_models import (
    DeveloperParametersCycleHCR,
    DeveloperParametersMerfish,
    DirOutputT,
    ExonExonJunctionBlockSizeT,
    General,
    GenomicRegions,
    PrimerCycleHCR,
    PrimerMerfish,
    ReadoutProbeCycleHCR,
    ReadoutProbeMerfish,
    SourceParamsCustom,
    SourceParamsEnsembl,
    SourceParamsNcbi,
    TargetProbeCycleHCR,
    TargetProbeMerfish,
)


class GenomicRegionBaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    dir_output: DirOutputT


class GenomicRegionGeneratorCustomConfig(GenomicRegionBaseConfig):
    dir_output: DirOutputT = "output_genomic_region_generator_custom"
    source: Annotated[
        Literal["custom"], Field(description="indicate which annotation should be used", default="custom")
    ]
    source_params: Annotated[
        SourceParamsCustom,
        Field(default_factory=SourceParamsCustom, description="Parameters for genome and gene annotation"),
    ]
    genomic_regions: Annotated[
        GenomicRegions,
        Field(
            default_factory=lambda: GenomicRegions(
                gene=True,
                intergenic=True,
                exon=True,
                exon_exon_junction=True,
                utr=True,
                cds=True,
                intron=True,
            ),
            description="Genomic regions that should be generated.",
        ),
    ]
    exon_exon_junction_block_size: ExonExonJunctionBlockSizeT


class GenomicRegionGeneratorEnsemblConfig(GenomicRegionBaseConfig):
    dir_output: DirOutputT = "output_genomic_region_generator_ensembl"
    source: Annotated[
        Literal["ensembl"], Field(description="indicate which annotation should be used", default="ensembl")
    ]
    source_params: Annotated[
        SourceParamsEnsembl,
        Field(default_factory=SourceParamsEnsembl, description="Parameters for genome and gene annotation"),
    ]
    genomic_regions: Annotated[
        GenomicRegions,
        Field(
            default_factory=lambda: GenomicRegions(
                gene=False,
                intergenic=False,
                exon=True,
                exon_exon_junction=False,
                utr=False,
                cds=False,
                intron=False,
            ),
            description="Genomic regions that should be generated.",
        ),
    ]
    exon_exon_junction_block_size: ExonExonJunctionBlockSizeT


class GenomicRegionGeneratorNcbiConfig(GenomicRegionBaseConfig):
    dir_output: DirOutputT = "output_genomic_region_generator_ncbi"
    source: Annotated[
        Literal["ncbi"], Field(description="indicate which annotation should be used", default="ncbi")
    ]
    source_params: Annotated[
        SourceParamsNcbi,
        Field(default_factory=SourceParamsNcbi, description="Parameters for genome and gene annotation"),
    ]
    genomic_regions: Annotated[
        GenomicRegions,
        Field(
            default_factory=lambda: GenomicRegions(
                gene=False,
                intergenic=False,
                exon=True,
                exon_exon_junction=False,
                utr=False,
                cds=False,
                intron=False,
            ),
            description="Genomic regions that should be generated.",
        ),
    ]
    exon_exon_junction_block_size: ExonExonJunctionBlockSizeT


class CycleHCRProbeDesignerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: PositiveInt
    general: Annotated[
        General,
        Field(
            default_factory=lambda: General(
                n_jobs=2, dir_output="output_cyclehcr_probe_designer", top_n_sets=3
            ),
            description="General parameters of the pipeline.",
        ),
    ]
    target_probe: TargetProbeCycleHCR
    readout_probe: ReadoutProbeCycleHCR
    primer: PrimerCycleHCR
    developer_param: DeveloperParametersCycleHCR


class MerfishProbeDesignerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: PositiveInt
    general: Annotated[
        General,
        Field(
            default_factory=lambda: General(
                n_jobs=4, dir_output="output_merfish_probe_designer", top_n_sets=3
            ),
            description="General parameters of the pipeline.",
        ),
    ]
    target_probe: TargetProbeMerfish
    readout_probe: ReadoutProbeMerfish
    primer: PrimerMerfish
    developer_param: DeveloperParametersMerfish
