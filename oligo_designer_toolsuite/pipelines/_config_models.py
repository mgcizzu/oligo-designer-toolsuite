from __future__ import annotations

from math import isclose
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveInt,
    PositiveInt,
    ValidationInfo,
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
    A: Annotated[PositiveInt | None, Field(default=None)]
    T: Annotated[PositiveInt | None, Field(default=None)]
    C: Annotated[PositiveInt | None, Field(default=None)]
    G: Annotated[PositiveInt | None, Field(default=None)]


FastaFileListT = Annotated[list[str], Field(min_length=1)]

FilesFastaDatabaseT = Annotated[
    FastaFileListT,
    Field(
        description="List of paths to FASTA file(s) containing sequences from which probes will be generated. These files should contain genomic regions of interest (e.g., exons, exon-exon junctions). Hint: use the genomic_region_generator pipeline to create FASTA files of genomic regions of interest."
    ),
]

FilesFastaReferenceDatabaseT = Annotated[
    FastaFileListT,
    Field(
        description="List of paths to FASTA file(s) containing sequences used as reference for the specificity filters. Hint: use the genomic_region_generator pipeline to create FASTA files of genomic regions of interest."
    ),
]

GCContentMinT = Annotated[float, Field(description="minimum GC content of oligos", ge=0, le=100)]
GCContentMaxT = Annotated[float, Field(description="maximum GC content of oligos", ge=0, le=100)]


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
    files_fasta_reference_database: FilesFastaReferenceDatabaseT

    isoform_consensus: Annotated[
        float,
        Field(
            description="min isoform consesnsus for probes, i.e. how many transcripts of the total number of transcripts of a gene are covered by the probe, given in %",
            ge=0,
            le=100,
        ),
    ]
    GC_content_min: GCContentMinT
    GC_content_max: GCContentMaxT
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


TmMinT = Annotated[NonNegativeFloat, Field(description="minimum melting temperature of oligos")]
TmOptT = Annotated[NonNegativeFloat, Field(description="optimal melting temperature of oligos")]
TmMaxT = Annotated[NonNegativeFloat, Field(description="maximum melting temperature of oligos")]

TSecondaryStructureT = Annotated[
    PositiveInt, Field(description=" Temperature at which the free energy is calculated")
]

WeightT = Annotated[float, Field(description="weight in the efficiency score  of the respective measure")]


class TargetProbeCycleHCR(TargetProbeBase):
    files_fasta_database: FilesFastaDatabaseT = [
        "data/genomic_regions/exon_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
        "data/genomic_regions/exon_exon_junction_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
    ]
    files_fasta_reference_database: FilesFastaReferenceDatabaseT = [
        "data/genomic_regions/gene_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna"
    ]

    isoform_consensus: float = 0
    GC_content_min: GCContentMinT = 30
    GC_content_max: GCContentMaxT = 90
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
    Tm_min: TmMinT = 90
    Tm_max: TmMaxT = 200
    T_secondary_structure: TSecondaryStructureT = 90
    junction_region_size: Annotated[
        NonNegativeInt,
        Field(
            default=13,
            description="size of the seed region around the junction site for blast seed region filter; set to 0 if junction region should not be considered for blast search",
        ),
    ]
    Tm_weight: WeightT = 1
    isoform_weight: WeightT = 10
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
        Literal["DNA_NN1", "DNA_NN2", "DNA_NN3", "DNA_NN4"], Field(default="DNA_NN3", description="default")
    ]
    tmm_table: Annotated[Literal["DNA_TMM1"], Field(default="DNA_TMM1", description="default")]
    imm_table: Annotated[Literal["DNA_IMM1"], Field(default="DNA_IMM1", description="default")]
    de_table: Annotated[Literal["DNA_DE1"], Field(default="DNA_DE1", description="default")]
    dnac1: Annotated[NonNegativeInt, Field(default=25, description="[nM]; default")]
    dnac2: Annotated[NonNegativeInt, Field(default=25, description="[nM]; default")]
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


class BlastnSearchParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    # use exclude_if=lambda v: v is None on a Field level here and not exlcude_none during model dumpimg,
    # because for other config arguments, the None is actually needed
    # but for blastn we don't want to provide default parameters but have the defaults handled directly
    # by blastn (as there can be quite complicated dependencies between arguments)
    # don't allow
    # -h
    # -help
    # -version
    # -query
    query_loc: Annotated[
        str | None,
        Field(
            default=None,
            alias="-query_loc",
            description="Location on the query sequence in 1-based offsets (Format: start-stop).",
            exclude_if=lambda v: v is None,
        ),
    ]
    strand: Annotated[
        Literal["plus", "minus", "both"] | None,
        Field(
            default=None,
            alias="-strand",
            description="Query strand(s) to search against database/subject. Choice of both, minus, or plus.",
            exclude_if=lambda v: v is None,
        ),
    ]
    task: Annotated[
        Literal["megablast", "dc-megablast", "blastn", "blastn-short", "rmblastn"] | None,
        Field(default=None, alias="-task", description="Supported tasks.", exclude_if=lambda v: v is None),
    ]
    # don't allow
    # -db
    # -out
    evalue: Annotated[
        float | None,
        Field(
            default=None,
            alias="-evalue",
            description="Expectation value (E) threshold for saving hits. Default = 10 (1000 for blastn-short)",
            exclude_if=lambda v: v is None,
        ),
    ]
    word_size: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-word_size",
            ge=4,
            description="Length of initial exact match.",
            exclude_if=lambda v: v is None,
        ),
    ]
    gapopen: Annotated[
        NonNegativeInt | None,
        Field(
            default=None, alias="-gapopen", description="Cost to open a gap.", exclude_if=lambda v: v is None
        ),
    ]
    gapextend: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-gapextend",
            description="Cost to extend a gap.",
            exclude_if=lambda v: v is None,
        ),
    ]
    penalty: Annotated[
        NonPositiveInt | None,
        Field(
            default=None,
            alias="-penalty",
            description="Penalty for a nucleotide mismatch.",
            exclude_if=lambda v: v is None,
        ),
    ]
    reward: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-reward",
            description="Reward for a nucleotide match.",
            exclude_if=lambda v: v is None,
        ),
    ]
    # don't allow use_index/index_name as another file would be needed
    # don't allow subject/subject_loc as another file would be needed
    # don't allow
    # -outfmt
    # -show_gis
    num_descriptions: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-num_descriptions",
            description="Number of database sequences to show one-line descriptions for.",
            exclude_if=lambda v: v is None,
        ),
    ]
    num_alignments: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-num_alignments",
            description="Number of database sequences to show alignments for.",
            exclude_if=lambda v: v is None,
        ),
    ]
    # don't allow
    # line_length
    # html
    sorthits: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-sorthits",
            le=4,
            description="Sorting option for hits.",
            exclude_if=lambda v: v is None,
        ),
    ]
    sorthsps: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-sorthsps",
            le=4,
            description="Sorting option for hps.",
            exclude_if=lambda v: v is None,
        ),
    ]
    dust: Annotated[
        str | None,
        Field(
            default=None,
            alias="-dust",
            description="Filter query sequence with dust.",
            exclude_if=lambda v: v is None,
        ),
    ]
    # don't allow
    # filtering_db as another file would be needed
    # window_masker_taxid
    # window_masker_db as then another file would be needed
    soft_masking: Annotated[
        bool | None,
        Field(
            default=None,
            alias="-soft_masking",
            description="Apply filtering locations as soft masks (i.e., only for finding initial matches).",
            exclude_if=lambda v: v is None,
        ),
    ]
    lcase_masking: Annotated[
        Literal[""] | None,
        Field(
            default=None,
            alias="-lcase_masking",
            description="Use lower case filtering in query and subject sequence(s).",
            exclude_if=lambda v: v is None,
        ),
    ]
    # don't allow the following options as additional files would be needed
    # gilist
    # seqidlist
    # negative_gilist
    # negative_seqidlist
    # taxids (theoretically no extra file needed, but we blast against only the target genome)
    # negative_taxids
    # taxidlist
    # negative_taxidlist
    # no_taxid_expansion
    # entrez_query
    db_soft_mask: Annotated[
        str | None,
        Field(
            default=None,
            alias="-db_soft_mask",
            description="Filtering algorithm ID to apply to the BLAST database as soft mask (i.e., only for finding initial matches).",
            exclude_if=lambda v: v is None,
        ),
    ]
    db_hard_mask: Annotated[
        str | None,
        Field(
            default=None,
            alias="-db_hard_mask",
            description="Filtering algorithm ID to apply to the BLAST database as hard mask (i.e., sequence is masked for all phases of search).",
            exclude_if=lambda v: v is None,
        ),
    ]
    perc_identity: Annotated[
        float | None,
        Field(
            default=None,
            alias="-perc_identity",
            description="Percent identity cutoff.",
            ge=0,
            le=100,
            exclude_if=lambda v: v is None,
        ),
    ]
    qcov_hsp_perc: Annotated[
        float | None,
        Field(
            default=None,
            alias="-qcov_hsp_perc",
            description="Percent query coverage per hsp.",
            ge=0,
            le=100,
            exclude_if=lambda v: v is None,
        ),
    ]
    max_hsps: Annotated[
        PositiveInt | None,
        Field(
            default=None,
            alias="-max_hsps",
            description="Set maximum number of HSPs per subject sequence to save for each query.",
            exclude_if=lambda v: v is None,
        ),
    ]
    culling_limit: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-culling_limit",
            description="If the query range of a hit is enveloped by that of at least this many higher-scoring hits, delete the hit",
            exclude_if=lambda v: v is None,
        ),
    ]
    best_hit_overhang: Annotated[
        float | None,
        Field(
            default=None,
            alias="-best_hit_overhang",
            description="Best Hit algorithm overhang value (recommended value: 0.1).",
            gt=0,
            lt=0.5,
            exclude_if=lambda v: v is None,
        ),
    ]
    best_hit_score_edge: Annotated[
        float | None,
        Field(
            default=None,
            alias="-best_hit_score_edge",
            description="Best Hit algorithm score edge value (recommended value: 0.1)",
            gt=0,
            lt=0.5,
            exclude_if=lambda v: v is None,
        ),
    ]
    subject_besthit: Annotated[
        Literal[""] | None,
        Field(
            default=None,
            alias="-subject_besthit",
            description="Turn on best hit per subject sequence.",
            exclude_if=lambda v: v is None,
        ),
    ]
    max_target_seqs: Annotated[
        PositiveInt | None,
        Field(
            default=None,
            alias="-max_target_seqs",
            description="Maximum number of aligned sequences to keep.",
            exclude_if=lambda v: v is None,
        ),
    ]
    template_type: Annotated[
        Literal["coding", "coding_and_optimal", "optimal"] | None,
        Field(
            default=None,
            alias="-template_type",
            description="Discontiguous MegaBLAST template type. Allowed values are coding, optimal and coding_and_optimal.",
            exclude_if=lambda v: v is None,
        ),
    ]
    # template_length is actually int, but only 3 values, therefore implemented as literal
    template_length: Annotated[
        Literal["16", "18", "21"] | None,
        Field(
            default=None,
            alias="-template_length",
            description="Discontiguous MegaBLAST template length.",
            exclude_if=lambda v: v is None,
        ),
    ]
    db_size: Annotated[
        int | None,
        Field(
            default=None,
            alias="-db_size",
            description="Effective length of the database.",
            exclude_if=lambda v: v is None,
        ),
    ]
    searchsp: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-searchsp",
            description="Effective length of the search space.",
            exclude_if=lambda v: v is None,
        ),
    ]
    # don't allow because extra file needed
    # import_search_strategy
    # export_search_strategy
    xdrop_ungap: Annotated[
        float | None,
        Field(
            default=None,
            alias="-xdrop_ungap",
            description="X-dropoff value (in bits) for ungapped extensions.",
            exclude_if=lambda v: v is None,
        ),
    ]
    xdrop_gap: Annotated[
        float | None,
        Field(
            default=None,
            alias="-xdrop_gap",
            description="X-dropoff value (in bits) for preliminary gapped extensions.",
            exclude_if=lambda v: v is None,
        ),
    ]
    xdrop_gap_final: Annotated[
        float | None,
        Field(
            default=None,
            alias="-xdrop_gap_final",
            description="X-dropoff value (in bits) for final gapped alignment.",
            exclude_if=lambda v: v is None,
        ),
    ]
    no_greedy: Annotated[
        Literal[""] | None,
        Field(
            default=None,
            alias="-no_greedy",
            description="Use non-greedy dynamic programming extension.",
            exclude_if=lambda v: v is None,
        ),
    ]
    min_raw_gapped_score: Annotated[
        int | None,
        Field(
            default=None,
            alias="-min_raw_gapped_score",
            description="Minimum raw gapped score to keep an alignment in the preliminary gapped and trace-back stages. Normally set based upon expect value.",
            exclude_if=lambda v: v is None,
        ),
    ]
    ungapped: Annotated[
        Literal[""] | None,
        Field(
            default=None,
            alias="-ungapped",
            description="Perform ungapped alignment only?",
            exclude_if=lambda v: v is None,
        ),
    ]
    window_size: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-window_size",
            description="Multiple hits window size, use 0 to specify 1-hit algorithm.",
            exclude_if=lambda v: v is None,
        ),
    ]
    off_diagonal_range: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            alias="-off_diagonal_range",
            description="Number of off-diagonals to search for the 2nd hit, use 0 to turn off.",
            exclude_if=lambda v: v is None,
        ),
    ]
    # don't allow
    # parse_deflines
    # num_threads
    # mt_mode
    # remote

    @field_validator("*")
    @classmethod
    def prevent_none(cls, v: Any, ctx: ValidationInfo) -> Any:
        assert v is not None, f"{ctx.field_name} can't be None"
        return v


class BlastnHitParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage: Annotated[
        float | None,
        Field(
            default=50,
            ge=0,
            le=100,
            description="alternatively, min_alignment_length can be used (ignored if min_alignment_length is set)",
        ),
    ]
    min_alignment_length: Annotated[
        NonNegativeInt | None,
        Field(
            default=None,
            description="alternatively, coverage can be used",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_enforce(cls, data: Any) -> Any:
        # Allow constructing from None / already validated model
        if data is None:
            return {}
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return data  # let pydantic raise its own error

        has_cov = "coverage" in data
        has_min = "min_alignment_length" in data

        # If both are explicitly provided -> error
        if (
            has_cov
            and has_min
            and data.get("coverage") is not None
            and data.get("min_alignment_length") is not None
        ):
            raise ValueError("Provide only one of 'coverage' or 'min_alignment_length', not both.")

        # Precedence rule: if min_alignment_length is provided (non-None) and coverage not explicitly provided,
        # then drop coverage (so default 50 won't conflict)
        if data.get("min_alignment_length") is not None and not has_cov:
            data = dict(data)
            data["coverage"] = None

        return data

    @model_validator(mode="after")
    def _check_mutually_exclusive(self) -> Self:
        if (self.coverage is None) == (self.min_alignment_length is None):
            raise ValueError("Exactly one of 'coverage' or 'min_alignment_length' must be set.")
        return self


SecondaryStructuresThresholdDeltaGT = Annotated[
    float,
    Field(
        default=0,
        description="threshold for the secondary structure free energy -> oligo rejected if it presents a structure with a negative free energy at the defined temperature",
    ),
]


class TargetProbeDev(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Tm_parameters: TmParameters | None
    Tm_chem_correction_parameters: Annotated[
        TmChemCorrectionParameters | None,
        Field(default=None, description="if chem correction desired, please add parameters below"),
    ]
    Tm_salt_correction_parameters: Annotated[
        TmSaltCorrectionParameters | None,
        Field(default=None, description="if salt correction desired, please add parameters below"),
    ]
    secondary_structures_threshold_deltaG: SecondaryStructuresThresholdDeltaGT
    specificity_blastn_search_parameters: BlastnSearchParameters
    specificity_blastn_hit_parameters: BlastnHitParameters
    cross_hybridization_blastn_search_parameters: BlastnSearchParameters
    cross_hybridization_blastn_hit_parameters: BlastnHitParameters


class TargetProbeDevCycleHCR(TargetProbeDev):
    Tm_parameters: Annotated[
        TmParameters | None, Field(default_factory=lambda: TmParameters(saltcorr=0), description="")
    ]

    specificity_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=100,
                strand="minus",
                word_size=10,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
                max_hsps=1000,
            )
        ),
    ]
    specificity_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(coverage=90))
    ]

    cross_hybridization_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=100,
                strand="minus",
                word_size=7,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
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
    target_probe: TargetProbeDevCycleHCR


############################################################
# Merfish
############################################################

LengthMinT = Annotated[NonNegativeInt, Field(description="mininum length of probe")]
LengthMaxT = Annotated[NonNegativeInt, Field(description="maximum length of probe")]


class TargetProbeMerfish(TargetProbeBase):
    files_fasta_database: list[str] = [
        "data/genomic_regions/exon_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
        "data/genomic_regions/exon_exon_junction_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
    ]
    files_fasta_reference_database: FilesFastaReferenceDatabaseT = [
        "data/genomic_regions/exon_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
        "data/genomic_regions/exon_exon_junction_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
    ]

    isoform_consensus: float = 50
    GC_content_min: GCContentMinT = 43
    GC_content_max: GCContentMaxT = 63
    homopolymeric_base_n: Annotated[
        HomopolymerThresholds, Field(default_factory=lambda: HomopolymerThresholds(A=6, T=6, C=6, G=6))
    ]

    set_size_min: PositiveInt = 50
    set_size_opt: PositiveInt = 50
    distance_between_target_probes: NonNegativeInt = 0
    n_sets: PositiveInt = 100

    GC_content_opt: Annotated[
        float, Field(default=53, description="optimal GC content of oligos", ge=0, le=100)
    ]

    length_min: LengthMinT = 30
    length_max: LengthMaxT = 30

    Tm_min: TmMinT = 66
    Tm_opt: TmOptT = 72
    Tm_max: TmMaxT = 76

    T_secondary_structure: TSecondaryStructureT = 76

    GC_weight: WeightT = 1
    Tm_weight: WeightT = 1
    isoform_weight: WeightT = 2


class BaseProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    A: Annotated[float, Field(default=0.25, ge=0, le=1)]
    C: Annotated[float, Field(default=0.25, ge=0, le=1)]
    G: Annotated[float, Field(default=0.25, ge=0, le=1)]
    T: Annotated[float, Field(default=0.25, ge=0, le=1)]

    @model_validator(mode="after")
    def _check_sums_up_to_1(self) -> Self:
        sum_probabilities = self.A + self.C + self.G + self.T
        if not isclose(sum_probabilities, 1):
            raise ValueError("The probabilities for all 4 bases needs to sum up to 1.")
        return self


class ReadoutProbeFish(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files_fasta_reference_database: Annotated[
        FilesFastaReferenceDatabaseT,
        Field(
            default=[
                "data/genomic_regions/exon_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
                "data/genomic_regions/exon_exon_junction_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
            ]
        ),
    ]
    length: Annotated[PositiveInt, Field(description="length of readout probes")]
    base_probabilities: Annotated[
        BaseProbabilities,
        Field(description="probabilities of each base for random readout probe sequence generation"),
    ]
    GC_content_min: GCContentMinT
    GC_content_max: GCContentMaxT
    homopolymeric_base_n: Annotated[
        HomopolymerThresholds,
        Field(
            default_factory=lambda: HomopolymerThresholds(G=3),
            description="minimum number of nucleotides to consider it a homopolymeric run per base",
        ),
    ]
    channels_ids: Annotated[
        list[str],
        Field(default=["Alexa488", "Cy3b", "Alexa647"], description="names of fluorescent channels"),
    ]


class OligoPropertyWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    length_oligo: Annotated[float | None, Field(default=None)]
    GC_content_oligo: Annotated[float | None, Field(default=None)]
    TmNN_oligo: Annotated[float | None, Field(default=None)]
    DG_secondary_structure_oligo: Annotated[float | None, Field(default=None)]
    length_selfcomplement_oligo: Annotated[float | None, Field(default=None)]


class ReadoutProbeMerfish(ReadoutProbeFish):

    length: PositiveInt = 20
    base_probabilities: Annotated[
        BaseProbabilities, Field(default_factory=lambda: BaseProbabilities(A=0.25, C=0.00, G=0.50, T=0.25))
    ]
    GC_content_min: GCContentMinT = 40
    GC_content_max: GCContentMaxT = 50
    set_size: Annotated[PositiveInt, Field(default=16, description="total number of readout probes")]
    homogeneous_properties_weights: Annotated[
        OligoPropertyWeights,
        Field(default_factory=lambda: OligoPropertyWeights(TmNN_oligo=1.0, GC_content_oligo=1.0)),
    ]
    n_bits: Annotated[PositiveInt, Field(default=16, description="number of bits contained in each barcode")]
    min_hamming_dist: Annotated[
        PositiveInt, Field(default=4, description="minimum distance between two valid barcodes")
    ]
    hamming_weight: Annotated[
        PositiveInt, Field(default=2, description="number of bits containing one in each barcode")
    ]


class PrimerFish(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files_fasta_reference_database: Annotated[
        FilesFastaReferenceDatabaseT,
        Field(
            default=[
                "data/genomic_regions/exon_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
                "data/genomic_regions/exon_exon_junction_annotation_source-NCBI_species-Homo_sapiens_annotation_release-110_genome_assemly-GRCh38.fna",
            ]
        ),
    ]
    reverse_primer_sequence: Annotated[
        DNA,
        Field(
            default="CCCTATAGTGAGTCGTATTA",
            description="defaults to reverse complement of 20 nt T7 promoter sequence, change if different sequence desired",
        ),
    ]
    length: Annotated[PositiveInt, Field(default=20, description="length of forward primer")]
    base_probabilities: Annotated[
        BaseProbabilities,
        Field(description="probabilities of each base for random primer sequence generation"),
    ]
    GC_content_min: GCContentMinT = 50
    GC_content_max: GCContentMaxT = 65
    number_GC_GCclamp: Annotated[
        NonNegativeInt,
        Field(
            default=1,
            description="the minimum number of G or C nucleotides required within the specified number of bases",
        ),
    ]
    number_three_prime_base_GCclamp: Annotated[
        NonNegativeInt,
        Field(default=2, description="the number of bases to consider from the 3' end of the sequence"),
    ]
    homopolymeric_base_n: Annotated[
        HomopolymerThresholds, Field(default_factory=lambda: HomopolymerThresholds(A=4, T=4, C=4, G=4))
    ]
    max_len_selfcomplement: Annotated[NonNegativeInt, Field(default=6)]
    max_len_complement_reverse_primer: Annotated[NonNegativeInt, Field(default=5)]
    Tm_min: TmMinT = 60
    Tm_max: TmMaxT = 75
    T_secondary_structure: TSecondaryStructureT = 76


class PrimerMerfish(PrimerFish):
    secondary_structures_threshold_deltaG: SecondaryStructuresThresholdDeltaGT


class TargetProbeDevMerfish(TargetProbeDev):
    Tm_parameters: Annotated[
        TmParameters,
        Field(default_factory=lambda: TmParameters(nn_table="DNA_NN4", dnac1=5, dnac2=0, Na=300)),
    ]

    specificity_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=80,
                strand="minus",
                word_size=10,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
                max_hsps=1000,
            )
        ),
    ]
    specificity_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(min_alignment_length=17))
    ]

    cross_hybridization_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=80,
                strand="minus",
                word_size=7,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
            )
        ),
    ]
    cross_hybridization_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(min_alignment_length=17))
    ]


class ReadoutProbeDevFish(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_num_sequences: Annotated[
        PositiveInt,
        Field(
            default=100000, description="if not enough readout probes can be generated, increase this number"
        ),
    ]
    specificity_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=100,
                strand="minus",
                word_size=7,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
                max_hsps=1000,
            )
        ),
    ]
    specificity_blastn_hit_parameters: BlastnHitParameters
    cross_hybridization_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=100,
                strand="minus",
                word_size=7,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
            )
        ),
    ]
    cross_hybridization_blastn_hit_parameters: BlastnHitParameters


class ReadoutProbeDevMerfish(ReadoutProbeDevFish):
    Tm_parameters: Annotated[
        TmParameters, Field(default_factory=lambda: TmParameters(nn_table="DNA_NN4", Na=300))
    ]
    Tm_chem_correction_parameters: Annotated[
        TmChemCorrectionParameters | None,
        Field(default=None, description="if chem correction desired, please add parameters below"),
    ]
    Tm_salt_correction_parameters: Annotated[
        TmSaltCorrectionParameters | None,
        Field(default=None, description="if salt correction desired, please add parameters below"),
    ]
    specificity_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(min_alignment_length=11))
    ]
    cross_hybridization_blastn_hit_parameters: Annotated[
        BlastnHitParameters, Field(default_factory=lambda: BlastnHitParameters(min_alignment_length=11))
    ]
    n_combinations: Annotated[
        PositiveInt,
        Field(
            default=100000,
            description="number of random combinations of readout probe sets to iterate through",
        ),
    ]


class PrimerDevFish(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_num_sequences: Annotated[
        PositiveInt,
        Field(
            default=100000, description="if not enough readout probes can be generated, increase this number"
        ),
    ]
    Tm_parameters: Annotated[
        TmParameters,
        Field(
            default_factory=lambda: TmParameters(
                nn_table="DNA_NN4",
                dnac1=250,
                dnac2=250,
                Na=300,
            )
        ),
    ]
    Tm_chem_correction_parameters: Annotated[
        TmChemCorrectionParameters | None,
        Field(default=None, description="if chem correction desired, please add parameters below"),
    ]
    Tm_salt_correction_parameters: Annotated[
        TmSaltCorrectionParameters | None,
        Field(default=None, description="if salt correction desired, please add parameters below"),
    ]
    specificity_refrence_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=100,
                strand="minus",
                word_size=7,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
                max_hsps=1000,
            )
        ),
    ]
    specificity_refrence_blastn_hit_parameters: Annotated[
        BlastnHitParameters,
        Field(
            default_factory=lambda: BlastnHitParameters(min_alignment_length=14),
            description="can be turned into coverage",
        ),
    ]
    specificity_hybridization_probes_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            default_factory=lambda: BlastnSearchParameters(
                perc_identity=100,
                strand="minus",
                word_size=7,
                dust="no",
                soft_masking=False,
                max_target_seqs=10,
                max_hsps=1000,
            )
        ),
    ]
    specificity_hybridization_probes_blastn_hit_parameters: Annotated[
        BlastnHitParameters,
        Field(
            default_factory=lambda: BlastnHitParameters(min_alignment_length=11),
            description="can be turned into coverage",
        ),
    ]


class DeveloperParametersMerfish(DeveloperParametersBase):
    target_probe: TargetProbeDevMerfish
    readout_probe: ReadoutProbeDevMerfish
    primer: PrimerDevFish
