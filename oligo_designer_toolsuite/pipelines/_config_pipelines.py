from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ._config_models import (
    DirOutputT,
    ExonExonJunctionBlockSizeT,
    GenomicRegions,
    SourceParamsCustom,
    SourceParamsEnsembl,
    SourceParamsNcbi,
)


class GenomicRegionGeneratorCustomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

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


class GenomicRegionGeneratorEnsemblConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

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


class GenomicRegionGeneratorNcbiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

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
