from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    RootModel,
    field_validator,
    model_validator,
)
from typing_extensions import Self

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
            default="data/annotations/custom_GCF_000001405.40_GRCh38.p14_genomic_chr16.gtf",
            description="GTF file with gene annotation",
        ),
    ]
    file_sequence: Annotated[
        str,
        Field(
            default="data/annotations/custom_GCF_000001405.40_GRCh38.p14_genomic_chr16.fna",
            description="FASTA file with genome sequence",
        ),
    ]
    files_source: Annotated[
        str | None, Field(default="NCBI", description="original source of the genomic files")
    ]
    species: Annotated[
        str | None,
        Field(default="Homo_sapiens", description="species of provided annotation, leave empty if unknown"),
    ]
    annotation_release: Annotated[
        str | None,
        Field(default="110", description="release number of provided annotation, leave empty if unknown"),
    ]
    genome_assembly: Annotated[
        str | None,
        Field(default="GRCh38", description="genome assembly of provided annotation, leave empty if unknown"),
    ]


class SourceParamsEnsembl(BaseModel):
    """
    Source parameters for the custom Genomic Region Generator
    """

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    species: Annotated[str, Field(default="homo_sapiens", description="species of provided annotation")]
    annotation_release: Annotated[
        str, Field(default="current", description="release number of provided annotation")
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
        Field(default="vertebrate_mammalian", description="taxon of the species"),
    ]
    species: Annotated[str, Field(default="Homo_sapiens", description="species of provided annotation")]
    annotation_release: Annotated[
        str, Field(default="110", description="release number of provided annotation")
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

DNA = Annotated[str, Field(pattern=r"^[ATGC]+$")]


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


class General(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_jobs: Annotated[
        PositiveInt,
        Field(
            description="number of cores used to run the pipeline and 2*n_jobs +1 of regions that should be stored in cache. If memory consumption of pipeline is too high reduce this number, if a lot of RAM is available increase this number to decrease runtime"
        ),
    ]
    dir_output: Annotated[
        str, Field(description="name of the directory where the output files will be written")
    ]
    write_intermediate_steps: Annotated[
        bool,
        Field(
            default=True,
            description="if true, writes the oligo sequences after each step of the pipeline into a csv file",
        ),
    ]
    top_n_sets: Annotated[
        PositiveInt,
        Field(description="maximum number of sets to report in *_probes.yaml and *_probes_order.yaml"),
    ]


class HomopolymerThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    A: PositiveInt
    T: PositiveInt
    C: PositiveInt
    G: PositiveInt


class TargetProbeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_regions: Annotated[
        str | None,
        Field(
            default="data/genes/custom_3.txt",
            description="file with a list the genes used to generate the probe sequences, leave empty if all the genes are used",
        ),
    ]
    files_fasta_database: Annotated[
        list[str],
        Field(
            description="fasta file with sequences form which the probes should be generated. Hint: use the genomic_region_generator pipeline to create fasta files of genomic regions of interest"
        ),
    ]
    files_fasta_reference_database: Annotated[
        list[str],
        Field(
            description="fasta file with sequences used as reference for the specificity filters. Hint: use the genomic_region_generator pipeline to create fasta files of genomic regions of interest"
        ),
    ]

    isoform_consensus: Annotated[
        float,
        Field(
            description="min isoform consesnsus for probes, i.e. how many transcripts of the total number of transcripts of a gene are covered by the probe, given in %",
            ge=0,
            le=100,
        ),
    ]
    GC_content_min: Annotated[float, Field(description="minimum GC content of oligos", ge=0, le=100)]
    GC_content_max: Annotated[float, Field(description="maximum GC content of oligos", ge=0, le=100)]
    homopolymeric_base_n: Annotated[
        HomopolymerThresholds,
        Field(description="minimum number of nucleotides to consider it a homopolymeric run per base"),
    ]

    set_size_min: Annotated[
        PositiveInt,
        Field(
            description="minimum size of probe sets (in case there exist no set of the optimal size) -> genes with less oligos will be filtered out and stored in regions_with_insufficient_oligos_for_db_probes"
        ),
    ]
    set_size_opt: Annotated[PositiveInt, Field(description="optimal size of probe sets")]
    distance_between_target_probes: Annotated[
        NonNegativeInt,
        Field(
            description="how much overlap should be allowed between oligos, e.g. if oligos can overlpap x bases choose -x, if oligos can be next to one another choose 0, if oligos should be x bases apart choose x"
        ),
    ]
    n_sets: Annotated[PositiveInt, Field(description="maximum number of sets to generate")]


class TargetProbeCycleHCR(TargetProbeBase):
    files_fasta_database: list[str] = [
        "data/genomic_regions/exon_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
        "data/genomic_regions/exon_exon_junction_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
    ]
    files_fasta_reference_database: list[str] = [
        "data/genomic_regions/gene_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna"
    ]

    isoform_consensus: float = 0
    GC_content_min: float = 30
    GC_content_max: float = 90
    homopolymeric_base_n: Annotated[
        HomopolymerThresholds, Field(default_factory=lambda: HomopolymerThresholds(A=6, T=6, C=6, G=6))
    ]

    set_size_min: PositiveInt = 10
    set_size_opt: PositiveInt = 25
    distance_between_target_probes: NonNegativeInt = 2
    n_sets: PositiveInt = 30

    L_probe_sequence_length: Annotated[
        PositiveInt,
        Field(
            default=45,
            description=" L + spacer + R sequence should equal the total probe length, e.g. 45 + 2 + 45 = 92",
        ),
    ]
    gap_sequence_length: Annotated[
        NonNegativeInt, Field(default=2, description="gap between L and R probe sequences")
    ]
    R_probe_sequence_length: Annotated[
        PositiveInt,
        Field(
            default=45,
            description="L + spacer + R sequence should equal the total probe length, e.g. 45 + 2 + 45 = 92",
        ),
    ]
    Tm_min: Annotated[PositiveInt, Field(default=90, description="minimum melting temperature of oligos")]
    Tm_max: Annotated[PositiveInt, Field(default=200, description="maximum melting temperature of oligos")]
    T_secondary_structure: Annotated[
        PositiveInt, Field(default=90, description=" Temperature at which the free energy is calculated")
    ]
    junction_region_size: Annotated[
        NonNegativeInt,
        Field(
            default=13,
            description="size of the seed region around the junction site for blast seed region filter; set to 0 if junction region should not be considered for blast search",
        ),
    ]
    Tm_weight: Annotated[
        float,
        Field(
            default=1,
            description="weight of the Tm of the probe in the efficiency score, where score in abs(Tm_max - Tm_probe)",
        ),
    ]
    isoform_weight: Annotated[
        float,
        Field(
            default=10,
            description="weight of the isoform consensus of the probe in the efficiency score, where score is between 0 and 1",
        ),
    ]
    linker_sequence: Annotated[
        DNA, Field(default="TT", description="linker sequence between readout probe and target sequence")
    ]


class ReadoutProbeCycleHCR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_readout_probe_table: Annotated[
        str,
        Field(
            default="data/functional_oligos/cycle_hcr_readout_probes_with_codebook.tsv",
            description='csv/tsv file containing the readout probe table with the columns "bit", "channel", "readout_probe_id", "readout_probe__sequence", "L/R"',
        ),
    ]
    file_codebook: Annotated[
        str | None,
        Field(
            default="data/functional_oligos/cycle_hcr_codebook.tsv",
            description='optional, file containing the codebook, where columns = "bits" and rows = "region_id" and the entries represent the bit encoding in 0/1 for the bits used for each region',
        ),
    ]

    @field_validator("file_readout_probe_table")
    @classmethod
    def must_be_csv_or_tsv(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.endswith(".csv") or v.endswith(".tsv")):
            raise ValueError("File must end with .csv or .tsv")
        return v


class PrimerCycleHCR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forward_primer_sequence: Annotated[
        DNA,
        Field(
            default="TAATACGACTCACTATAGCGTCATC",
            description="defaults to T7 promoter sequence, change if different sequence desired",
        ),
    ]
    reverse_primer_sequence: Annotated[
        DNA,
        Field(
            default="CGACACCGAACGTGCGACAA",
            description="defaults to oligo sequence, change if different sequence desired",
        ),
    ]


class OligoSetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_graph_size: Annotated[
        PositiveInt,
        Field(
            default=5000,
            description="maximum number of oligos that are taken into consideration in the last step (5000 -> ~5GB, 2500 -> ~1GB)",
        ),
    ]
    n_attempts: Annotated[
        PositiveInt, Field(default=100000, description="number of attempts to find the optimal set of oligos")
    ]
    heuristic: Annotated[
        bool,
        Field(
            default=True,
            description="apply heuristic pre-search to reduce search space and runtime of oligo set selection",
        ),
    ]
    heuristic_n_attempts: Annotated[
        PositiveInt,
        Field(
            default=100,
            description="number of attempts to find the optimal set of oligos for heuristic pre-search",
        ),
    ]


class TmParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # defaults are from Bio.SeqUtils.MeltingTemp.Tm_NN
    nn_table: Annotated[
        Literal["DNA_NN1", "DNA_NN4", "DNA_NN3", "DNA_NN4"], Field(default="DNA_NN3", description="default")
    ]
    tmm_table: Annotated[Literal["DNA_TMM1"], Field(default="DNA_TMM1", description="default")]
    imm_table: Annotated[Literal["DNA_IMM1"], Field(default="DNA_IMM1", description="default")]
    de_table: Annotated[Literal["DNA_DE1"], Field(default="DNA_DE1", description="default")]
    dnac1: Annotated[PositiveInt, Field(default=25, description="[nM]; default")]
    dnac2: Annotated[PositiveInt, Field(default=25, description="[nM]; default")]
    saltcorr: Annotated[
        NonNegativeInt,
        Field(
            default=5,
            ge=0,
            le=7,
            description="salt correction method, see Bio.SeqUtils.MeltingTemp.salt_correction",
        ),
    ]
    Na: Annotated[NonNegativeInt, Field(default=50, description="[mM]; default")]
    K: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    Tris: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    Mg: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    dNTPs: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]


class TmChemCorrectionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # defaults are from Bio.SeqUtils.MeltingTemp.chem_correction
    DMSO: Annotated[float, Field(default=0, ge=0, le=100, description="Percent DMSO")]
    DMSOfactor: Annotated[
        float, Field(default=0.75, description="How much Tm should decrease per percent DMSO")
    ]
    fmd: Annotated[
        float,
        Field(default=0, description="Formamide concentration in %(fmdmethod=1) or molar (fmdmethod=2)."),
    ]
    fmdfactor: Annotated[
        float, Field(default=0.65, description="How much Tm should decrease per percent formamide")
    ]
    fmdmethod: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            le=2,
            description="Tm = Tm - factor(%formamide) (Default); Tm = Tm + (0.453(f(GC)) - 2.88) x [formamide]",
        ),
    ]
    GC: Annotated[float | None, Field(default=None, ge=0, le=100, description="GC content in percent.")]

    @model_validator(mode="after")
    def _check_fmd_vs_method(self) -> Self:
        # method 1: fmd is percent
        if self.fmdmethod == 1:
            if not (0 <= self.fmd <= 100):
                raise ValueError("For fmdmethod=1, fmd must be a percentage in [0, 100].")

        # method 2: fmd is molar, and GC is required by the formula
        elif self.fmdmethod == 2:
            if self.fmd < 0:
                raise ValueError("For fmdmethod=2, fmd must be a non-negative molar concentration.")
            if self.GC is None:
                raise ValueError("For fmdmethod=2, GC must be provided (0–100%) for the formula.")

        return self


class TmSaltCorrectionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # defaults are from Bio.SeqUtils.MeltingTemp.salt_correction
    Na: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    K: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    Tris: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    Mg: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    dNTPs: Annotated[NonNegativeInt, Field(default=0, description="[mM]; default")]
    method: Annotated[
        PositiveInt,
        Field(
            default=1,
            ge=1,
            le=7,
            description="Correction method to be applied. Methods 1-4 correct Tm, method 5 corrects deltaS, methods 6 and 7 correct 1/Tm.",
        ),
    ]


class ArgvList(RootModel[list[str]]):
    """
    Arguments for command line tool; key-value pairs are separate entries
    in a list, flags are entries as well, e.g. ["-perc_identity", "100", "-strand", "minus", "-ungapped"].
    Coerces scalars to str and forbids nested lists/dicts.
    """

    root: list[str]

    @model_validator(mode="before")
    @classmethod
    def coerce_flat_scalars(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("Expected a list of arguments (also for key-value pairs).")
        out: list[str] = []
        for i, item in enumerate(v):
            if isinstance(item, (list, dict)):
                raise ValueError(f"Argument entry at index {i} must be a scalar, got {type(item).__name__}")
            # bool -> "true"/"false", everything else -> str()
            if isinstance(item, bool):
                out.append("true" if item else "false")
            else:
                out.append(str(item))
        return out


class BlastnHitParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage: Annotated[
        float | None,
        Field(default=None, ge=0, le=100, description="alternatively, min_alignment_length can be used"),
    ]
    min_alignment_length: Annotated[
        NonNegativeInt | None, Field(default=None, description="alternatively, coverage can be used")
    ]

    @model_validator(mode="after")
    def check_mutually_exclusive(self) -> Self:
        """Ensure exactly one of coverage or min_alignment_length is provided."""
        if (self.coverage is None) == (self.min_alignment_length is None):
            # both None OR both set -> invalid
            raise ValueError("Exactly one of 'coverage' or 'min_alignment_length' must be set.")
        # remove the parameter that is None because the code downstream expects only
        # one parameter
        if self.coverage is None:
            delattr(self, "coverage")
        else:
            delattr(self, "min_alignment_length")
        return self


class TargetProbeDevCycleHCR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Tm_parameters: Annotated[TmParameters, Field(default_factory=lambda: TmParameters(saltcorr=0))]

    Tm_chem_correction_parameters: Annotated[
        TmChemCorrectionParameters | None,
        Field(default=None, description="if chem correction desired, please add parameters below"),
    ]
    Tm_salt_correction_parameters: Annotated[
        TmSaltCorrectionParameters | None,
        Field(default=None, description="if salt correction desired, please add parameters below"),
    ]
    secondary_structures_threshold_deltaG: Annotated[
        float,
        Field(
            default=0,
            description="threshold for the secondary structure free energy -> oligo rejected if it presents a structure with a negative free energy at the defined temperature",
        ),
    ]

    specificity_blastn_search_parameters: Annotated[
        ArgvList,
        Field(
            default_factory=lambda: ArgvList(
                root=[
                    "-perc_identity",
                    "100",
                    "-strand",
                    "minus",
                    "-word_size",
                    "10",
                    "-dust",
                    "no",
                    "-soft_masking",
                    "false",
                    "-max_target_seqs",
                    "10",
                    "-max_hsps",
                    "1000",
                ]
            )
        ),
    ]
    specificity_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(coverage=90))
    ]

    cross_hybridization_blastn_search_parameters: Annotated[
        ArgvList,
        Field(
            default_factory=lambda: ArgvList(
                root=[
                    "-perc_identity",
                    "100",
                    "-strand",
                    "minus",
                    "-word_size",
                    "7",
                    "-dust",
                    "no",
                    "-soft_masking",
                    "false",
                    "-max_target_seqs",
                    "10",
                ]
            )
        ),
    ]
    cross_hybridization_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(coverage=90))
    ]


class DeveloperParametersBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oligo_set_selection: OligoSetSelection


class DeveloperParametersCycleHCR(DeveloperParametersBase):
    model_config = ConfigDict(extra="forbid")

    target_probe: TargetProbeDevCycleHCR
