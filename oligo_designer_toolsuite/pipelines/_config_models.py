from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

# ---------------------------
# Genomic region generator options
# ---------------------------

DirOutputT = Annotated[
    str, Field(description="Name of the directory where the output files will be written.")
]

# separate source params definitions because the combination of fields
# and which are optional are different across custom/NCBI/Ensembl


class SourceParamsCustom(BaseModel):
    """
    Source parameters for the custom Genomic Region Generator
    """

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    file_annotation: Annotated[
        str,
        Field(
            description="GTF file with gene annotation",
            default="data/annotations/custom_GCF_000001405.40_GRCh38.p14_genomic_chr16.gtf",
        ),
    ]
    file_sequence: Annotated[
        str,
        Field(
            description="FASTA file with genome sequence",
            default="data/annotations/custom_GCF_000001405.40_GRCh38.p14_genomic_chr16.fna",
        ),
    ]
    files_source: Annotated[
        str | None, Field(description="original source of the genomic files", default="NCBI")
    ]
    species: Annotated[
        str | None,
        Field(description="species of provided annotation, leave empty if unknown", default="Homo_sapiens"),
    ]
    annotation_release: Annotated[
        int | str | None,
        Field(description="release number of provided annotation, leave empty if unknown", default=110),
    ]
    genome_assembly: Annotated[
        str | None,
        Field(description="genome assembly of provided annotation, leave empty if unknown", default="GRCh38"),
    ]


class SourceParamsEnsembl(BaseModel):
    """
    Source parameters for the custom Genomic Region Generator
    """

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    species: Annotated[str, Field(description="species of provided annotation", default="homo_sapiens")]
    annotation_release: Annotated[
        str, Field(description="release number of provided annotation", default="current")
    ]


class SourceParamsNcbi(BaseModel):
    """
    Source parameters for the custom Genomic Region Generator
    """

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    taxon: Annotated[
        Literal[
            "archaea",
            "bacteria",
            "fungi",
            "invertebrate",
            "mitochondrion",
            "plant",
            "plasmid",
            "plastid",
            "protozoa",
            "vertebrate_mammalian",
            "vertebrate_other",
            "viral",
        ],
        Field(description="taxon of the species", default="vertebrate_mammalian"),
    ]
    species: Annotated[str, Field(description="species of provided annotation", default="Homo_sapiens")]
    annotation_release: Annotated[
        int | str, Field(description="release number of provided annotation", default=110)
    ]


ExonExonJunctionBlockSizeT = Annotated[
    int,
    Field(
        default=50,
        ge=1,
        description=(
            "Block size (bp) around each exon–exon junction, i.e. +/- this many "
            "bp around the junction. It does not make sense to set this larger "
            "than the maximum oligo length."
        ),
    ),
]


class GenomicRegions(BaseModel):
    """
    Selection flags for which genomic regions to generate.
    Mirrors the `genomic_regions` mapping in the YAML configs.
    """

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    gene: Annotated[bool, Field(description="Generate gene regions.")]
    intergenic: Annotated[bool, Field(description="Generate intergenic regions.")]
    exon: Annotated[bool, Field(description="Generate exon regions.")]
    exon_exon_junction: Annotated[bool, Field(description="Generate exon–exon junction regions.")]
    utr: Annotated[bool, Field(description="Generate UTR regions.")]
    cds: Annotated[bool, Field(description="Generate coding sequence (CDS) regions.")]
    intron: Annotated[bool, Field(description="Generate intron regions.")]
