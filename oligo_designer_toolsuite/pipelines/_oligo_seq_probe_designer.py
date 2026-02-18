############################################
# imports
############################################

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

import yaml

from oligo_designer_toolsuite.database import OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_efficiency_filter import (
    AverageSetScoring,
    IsoformConsensusScorer,
    NormalizedDeviationFromOptimalGCContentScorer,
    NormalizedDeviationFromOptimalTmScorer,
    OligoScoring,
    OverlapTargetedExonsScorer,
    UniformDistanceScorer,
)
from oligo_designer_toolsuite.oligo_property_calculator import (
    GCContentProperty,
    IsoformConsensusProperty,
    LengthProperty,
    LengthSelfComplementProperty,
    NumTargetedTranscriptsProperty,
    PropertyCalculator,
    ReverseComplementSequenceProperty,
    ShortenedSequenceProperty,
    TargetedExonsProperty,
    TmNNProperty,
)
from oligo_designer_toolsuite.oligo_property_filter import (
    BasePropertyFilter,
    GCContentFilter,
    HardMaskedSequenceFilter,
    HomopolymericRunsFilter,
    MeltingTemperatureNNFilter,
    ProhibitedSequenceFilter,
    PropertyFilter,
    SecondaryStructureFilter,
    SelfComplementFilter,
    SoftMaskedSequenceFilter,
)
from oligo_designer_toolsuite.oligo_selection import IndependentSetsOligoSelection
from oligo_designer_toolsuite.oligo_specificity_filter import (
    BaseSpecificityFilter,
    BlastNFilter,
    CrossHybridizationFilter,
    ExactMatchFilter,
    RemoveAllFilterPolicy,
    RemoveByLargerRegionFilterPolicy,
    SpecificityFilter,
    VariantsFilter,
)
from oligo_designer_toolsuite.pipelines._utils import (
    base_log_parameters,
    base_parser,
    check_content_oligo_database,
    get_highly_abundant_kmer_sequences,
    pipeline_step_basic,
    preprocess_tm_parameters,
    setup_logging,
)
from oligo_designer_toolsuite.sequence_generator import OligoSequenceGenerator

############################################
# Oligo-Seq Probe Designer
############################################


class OligoSeqProbeDesigner:
    """
    A class for designing hybridization probes for Oligo-seq RNA detection assays.

    This class implements a complete pipeline for designing oligonucleotide probes compatible with the **Oligo-seq** method,
    a targeted RNA detection and sequencing approach that combines hybridization-based capture with next-generation sequencing (NGS)
    to profile gene expression at single-cell or subcellular resolution.

    **Oligo-seq Pipeline Overview:**
    1. **Target Probe Design**: Design gene-specific targeting sequences (26-30 nt) that bind to RNA transcripts
    2. **Output Generation**: Generate output files in multiple formats (TSV, YAML)

    Overview
    --------
    **Oligo-seq** (Oligonucleotide sequencing hybridization) is a novel RNA detection tool that merges *in situ hybridization* and *sequencing-based readout*.
    It allows multiplexed, quantitative RNA detection from extremely low input material (as few as ~50 cells) with high reproducibility and specificity. By
    focusing on *targeted exonic regions* and *exon–intron junctions*, Oligo-seq achieves robust capture of nascent and mature transcripts, enabling fine-grained
    resolution of gene expression states in different cell types (e.g., ESCs, XEN, and liver cells).

    Probe Structure
    ---------------
    **Oligo-seq Probes**
    - Single-stranded DNA oligonucleotides designed to hybridize directly to target RNA sequences.
    - Each probe consists of:
        - A **26-30nt targeting sequence** complementary to the RNA transcript.
    - A standard library includes ~92,000 probes covering ~1,800 genes, with an average of ~50 probes per gene.

    References
    ----------

    :ivar dir_output: Directory where probe design and library files will be written.
    :type dir_output: str
    :ivar write_intermediate_steps: Whether to save intermediate design and validation files (default: False).
    :type write_intermediate_steps: bool
    :ivar n_jobs: Number of parallel threads to use for probe design and computational validation.
    :type n_jobs: int
    """

    def __init__(self, write_intermediate_steps: bool, dir_output: str, n_jobs: int) -> None:
        """Constructor for the OligoSeqProbeDesigner class."""

        # create the output folder
        self.dir_output = os.path.abspath(dir_output)
        Path(self.dir_output).mkdir(parents=True, exist_ok=True)

        # setup logger
        setup_logging(
            dir_output=self.dir_output,
            pipeline_name="oligoseq_probe_designer",
            log_start_message=True,
        )

        ##### set class parameters #####
        self.write_intermediate_steps = write_intermediate_steps
        self.n_jobs = n_jobs

    def design_target_probes(
        self,
        # Step 1: Create Database Parameters
        target_probe_design_parameters: dict,
        # Step 2: Property Filter Parameters
        target_probe_isoform_consensus_filter: dict,
        target_probe_targeted_exons_filter: dict,
        target_probe_hard_masked_sequences_filter: dict,
        target_probe_soft_masked_sequences_filter: dict,
        target_probe_homopolymeric_runs_filter: dict,
        target_probe_GC_content_filter: dict,
        target_probe_prohibited_sequences_filter: dict,
        target_probe_self_complementarity_filter: dict,
        target_probe_melting_temperature_filter: dict,
        target_probe_secondary_structure_filter: dict,
        # Step 3: Specificity Filter Parameters
        target_probe_read_length_bias_filter: dict,
        target_probe_cross_hybridization_filter: dict,
        target_probe_specificity_blastn_filter: dict,
        target_probe_variant_filter: dict,
        # Step 4: Probe Scoring and Set Selection Parameters
        target_probe_set_selection_parameters: dict,
    ) -> OligoDatabase:
        """
        Design target probes for Oligo-seq experiments through a multi-step pipeline.

        This method performs the complete target probe design process, which includes:
        1. Creating an initial oligo database from input FASTA files using a sliding window approach
           with region splitting capability
        2. Filtering probes based on sequence properties (GC content, melting temperature, homopolymeric
           runs, secondary structure, self-complementarity)
        3. Filtering probes based on specificity to remove off-target binding, cross-hybridization,
           and variants using BLASTN or Bowtie searches, and hybridization probability filtering
        4. Organizing filtered probes into optimal sets based on weighted scoring criteria (targeted
           exons, isoform consensus, GC content, melting temperature) and distance constraints

        The resulting probes are gene-specific targeting sequences (typically 26-30 nt) that bind to
        RNA transcripts. These probes are used directly for hybridization-based capture in Oligo-seq
        experiments, which combine in situ hybridization with next-generation sequencing readout.

        **Step 1: Create Database Parameters**

        :param target_probe_design_parameters: Design parameters. Keys: ``region_ids`` (or from file),
            ``files_fasta_probe_database``, ``probe_length_min``, ``probe_length_max``, ``probe_split_region``.
        :type target_probe_design_parameters: dict

        **Step 2: Property Filter Parameters**

        :param target_probe_isoform_consensus_filter: Filter config. Keys: ``enabled``, ``isoform_consensus``.
        :type target_probe_isoform_consensus_filter: dict
        :param target_probe_targeted_exons_filter: Filter config. Keys: ``enabled``, ``targeted_exons``.
        :type target_probe_targeted_exons_filter: dict
        :param target_probe_hard_masked_sequences_filter: Filter config. Key: ``enabled``.
        :type target_probe_hard_masked_sequences_filter: dict
        :param target_probe_soft_masked_sequences_filter: Filter config. Key: ``enabled``.
        :type target_probe_soft_masked_sequences_filter: dict
        :param target_probe_homopolymeric_runs_filter: Filter config. Keys: ``enabled``, ``homopolymeric_base_n``.
        :type target_probe_homopolymeric_runs_filter: dict
        :param target_probe_GC_content_filter: Filter config. Keys: ``enabled``, ``GC_content_min``, ``GC_content_max``.
        :type target_probe_GC_content_filter: dict
        :param target_probe_prohibited_sequences_filter: Filter config. Keys: ``enabled``, ``prohibited_sequences``,
            optionally ``kmer_abundance_threshold``.
        :type target_probe_prohibited_sequences_filter: dict
        :param target_probe_self_complementarity_filter: Filter config. Keys: ``enabled``, ``max_len_selfcomplement``.
        :type target_probe_self_complementarity_filter: dict
        :param target_probe_melting_temperature_filter: Filter config. Keys: ``enabled``, ``Tm_min``, ``Tm_max``,
            ``Tm_parameters``, ``Tm_chem_correction_parameters``, ``Tm_salt_correction_parameters``.
        :type target_probe_melting_temperature_filter: dict
        :param target_probe_secondary_structure_filter: Filter config. Keys: ``enabled``, ``T``, ``thr_DG``.
        :type target_probe_secondary_structure_filter: dict

        **Step 3: Specificity Filter Parameters**

        :param target_probe_read_length_bias_filter: Filter config. Keys: ``enabled``, ``read_length_bias``.
        :type target_probe_read_length_bias_filter: dict
        :param target_probe_cross_hybridization_filter: Filter config. Keys: ``enabled``,
            ``cross_hybridization_search_parameters``, ``cross_hybridization_hit_parameters``.
        :type target_probe_cross_hybridization_filter: dict
        :param target_probe_specificity_blastn_filter: Filter config. Keys: ``enabled``,
            ``specificity_blastn_search_parameters``, ``specificity_blastn_hit_parameters``,
            ``files_fasta_reference_database``.
        :type target_probe_specificity_blastn_filter: dict
        :param target_probe_variant_filter: Filter config. Keys: ``enabled``, ``files_vcf_reference_database``,
            ``action`` ("flag" or "remove").
        :type target_probe_variant_filter: dict

        **Step 4: Probe Scoring and Set Selection Parameters**

        :param target_probe_set_selection_parameters: Set selection config. Keys: ``n_sets``, ``set_size_min``,
            ``set_size_opt``, ``distance_between_target_probes``, ``uniform_distance_weight``, ``isoform_weight``,
            ``targeted_exons_weight``, ``targeted_exons``, ``GC_weight``, ``GC_content_opt``, ``Tm_weight``, ``Tm_opt``,
            ``n_attempts_graph``, ``n_attempts_clique_enum``, ``diversification_fraction``, ``jaccard_opt``,
            ``jaccard_step``; ``Tm_parameters``, ``Tm_chem_correction_parameters``, ``Tm_salt_correction_parameters``,
            ``GC_content_min``, ``GC_content_max``, ``Tm_min``, ``Tm_max`` are injected by preprocessing.
        :type target_probe_set_selection_parameters: dict
        :return: An `OligoDatabase` object containing the designed target probes organized into sets.
            The database includes probe sequences, properties, and set assignments for each target gene.
        :rtype: OligoDatabase
        """

        target_probe_designer = TargetProbeDesigner(self.dir_output, self.n_jobs)

        oligo_database: OligoDatabase = target_probe_designer.create_oligo_database(
            region_ids=target_probe_design_parameters["region_ids"],
            oligo_length_min=target_probe_design_parameters["probe_length_min"],
            oligo_length_max=target_probe_design_parameters["probe_length_max"],
            split_region=target_probe_design_parameters["probe_split_region"],
            files_fasta_oligo_database=target_probe_design_parameters["files_fasta_probe_database"],
            min_oligos_per_gene=target_probe_set_selection_parameters["set_size_min"],
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="1_db_probes_initial")
            logging.info(f"Saved probe database for step 1 (Create Database) in directory {dir_database}")

        # Add highly abundant k-mers to prohibited sequences
        if target_probe_prohibited_sequences_filter["enabled"]:
            if target_probe_prohibited_sequences_filter["kmer_abundance_threshold"]:
                prohibited_sequences = get_highly_abundant_kmer_sequences(
                    files_fasta=target_probe_design_parameters["files_fasta_probe_database"],
                    kmer_abundance_threshold=target_probe_prohibited_sequences_filter[
                        "kmer_abundance_threshold"
                    ],
                )
            target_probe_prohibited_sequences_filter["prohibited_sequences"] = list(
                set(target_probe_prohibited_sequences_filter["prohibited_sequences"] + prohibited_sequences)
            )

        oligo_database = target_probe_designer.filter_by_property(
            oligo_database=oligo_database,
            target_probe_isoform_consensus_filter=target_probe_isoform_consensus_filter,
            target_probe_targeted_exons_filter=target_probe_targeted_exons_filter,
            target_probe_hard_masked_sequences_filter=target_probe_hard_masked_sequences_filter,
            target_probe_soft_masked_sequences_filter=target_probe_soft_masked_sequences_filter,
            target_probe_homopolymeric_runs_filter=target_probe_homopolymeric_runs_filter,
            target_probe_GC_content_filter=target_probe_GC_content_filter,
            target_probe_prohibited_sequences_filter=target_probe_prohibited_sequences_filter,
            target_probe_self_complementarity_filter=target_probe_self_complementarity_filter,
            target_probe_melting_temperature_filter=target_probe_melting_temperature_filter,
            target_probe_secondary_structure_filter=target_probe_secondary_structure_filter,
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="2_db_probes_property_filter")
            logging.info(f"Saved probe database for step 2 (Property Filters) in directory {dir_database}")

        oligo_database = target_probe_designer.filter_by_specificity(
            oligo_database=oligo_database,
            target_probe_read_length_bias_filter=target_probe_read_length_bias_filter,
            target_probe_cross_hybridization_filter=target_probe_cross_hybridization_filter,
            target_probe_specificity_blastn_filter=target_probe_specificity_blastn_filter,
            target_probe_variant_filter=target_probe_variant_filter,
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="3_db_probes_specificity_filter")
            logging.info(f"Saved probe database for step 3 (Specificity Filters) in directory {dir_database}")

        oligo_database = target_probe_designer.create_oligo_sets(
            oligo_database=oligo_database,
            target_probe_set_selection_parameters=target_probe_set_selection_parameters,
        )

        # Calculate oligo length, GC content, Tm, num targeted transcripts, isoform consensus, and length self complement
        length_property = LengthProperty()
        gc_content_property = GCContentProperty()
        TmNN_property = TmNNProperty(
            Tm_parameters=target_probe_set_selection_parameters["Tm_parameters"],
            Tm_chem_correction_parameters=target_probe_set_selection_parameters[
                "Tm_chem_correction_parameters"
            ],
            Tm_salt_correction_parameters=target_probe_set_selection_parameters[
                "Tm_salt_correction_parameters"
            ],
        )
        num_targeted_transcripts_property = NumTargetedTranscriptsProperty()
        isoform_consensus_property = IsoformConsensusProperty()
        length_self_complement_property = LengthSelfComplementProperty()
        calculator = PropertyCalculator(
            properties=[
                length_property,
                gc_content_property,
                TmNN_property,
                num_targeted_transcripts_property,
                isoform_consensus_property,
                length_self_complement_property,
            ]
        )
        oligo_database = calculator.apply(
            oligo_database=oligo_database, sequence_type="oligo", n_jobs=self.n_jobs
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="4_db_probes_probesets")
            logging.info(f"Saved probe database for step 4 (Specificity Filters) in directory {dir_database}")

        return oligo_database

    def generate_output(
        self,
        oligo_database: OligoDatabase,
        output_properties: list[str] | None = None,
    ) -> None:
        """
        Generate the final output files for the Oligo-seq probe design pipeline.

        This method writes all output files required for the Oligo-seq experiment, including probe
        sequences and properties in multiple formats. The output files are written to the pipeline's
        output directory.

        **Generated Output Files:**

        1. **oligo_seq_probes.yml**: Complete probe information in YAML format, including all specified
           properties for each probe in the top N sets per region.

        2. **oligo_seq_probes.tsv**: Complete probe information in TSV format, including all specified
           properties for each probe in the top N sets per region.

        3. **oligo_seq_probes.xlsx**: Complete probe information in Excel format with one sheet per region.
           Each sheet contains probe sets for that region with all specified properties.

        4. **oligo_seq_probes_order.yml**: Simplified YAML file containing only the essential sequences
           needed for ordering probes (oligo sequences).

        :param oligo_database: The `OligoDatabase` instance containing the final target probes
            with all sequences and properties. This should be the result of the `design_target_probes`
            method.
        :type oligo_database: OligoDatabase
        :param output_properties: List of property names to include in the output files. If None, a default
            set of properties will be included. Available properties include: 'source', 'species', 'gene_id',
            'chromosome', 'start', 'end', 'strand', 'oligo', 'target', 'length_oligo', 'GC_content_oligo',
            'TmNN_oligo', 'SNP_filter', 'isoform_consensus', 'length_selfcomplement_oligo', etc.
        :type output_properties: list[str] | None
        :return: None. All output files are written to the pipeline's output directory.
        """

        if output_properties is None:
            output_properties = [
                "source",
                "species",
                "annotation_release",
                "genome_assembly",
                "gene_id",
                "chromosome",
                "start",
                "end",
                "strand",
                "regiontype",
                "transcript_id",
                "exon_number",
                "oligo",
                "target",
                "length_oligo",
                "GC_content_oligo",
                "TmNN_oligo",
                "SNP_filter",
                "isoform_consensus",
                "length_selfcomplement_oligo",
            ]

        oligo_database.write_oligosets_to_yaml(
            properties=output_properties,
            ascending=True,
            filename="oligo_seq_probes",
        )

        oligo_database.write_oligosets_to_table(
            properties=output_properties,
            ascending=True,
            filename="oligo_seq_probes",
        )

        oligo_database.write_ready_to_order_yaml(
            properties=["oligo"],
            ascending=True,
            filename="oligo_seq_probes_order",
        )


############################################
# Oligo-Seq Target Probe Designer
############################################


class TargetProbeDesigner:
    """
    A class for designing target probes for Oligo-seq experiments through a multi-step pipeline.

    This class provides methods for the complete target probe design process, which includes:
    1. Creating an initial oligo database from input FASTA files using a sliding window approach
       with region splitting for memory efficiency
    2. Filtering probes based on sequence properties (GC content, melting temperature, homopolymeric
       runs, self-complementarity, secondary structure)
    3. Filtering probes based on specificity to remove off-target binding, cross-hybridization,
       variants, and high hybridization probability using BLASTN or Bowtie searches, and read length
       bias filtering
    4. Organizing filtered probes into optimal sets based on weighted scoring criteria (targeted
       exons, isoform consensus, GC content, melting temperature) and distance constraints

    The resulting probes are gene-specific targeting sequences (typically 26-30 nt) that bind to
    RNA transcripts. These probes are used directly for hybridization-based capture in Oligo-seq
    experiments, which combine in situ hybridization with next-generation sequencing readout. The
    probes target exonic regions and exon-intron junctions to capture both nascent and mature
    transcripts.

    :param dir_output: Directory path where output files will be saved.
    :type dir_output: str
    :param n_jobs: Number of parallel jobs to use for processing. Set to 1 for serial processing or higher
        values for parallel processing. This affects the parallelization of filtering, property calculation,
        and set generation operations.
    :type n_jobs: int
    """

    def __init__(self, dir_output: str, n_jobs: int) -> None:
        """Constructor for the TargetProbeDesigner class."""

        ##### create the output folder #####
        self.dir_output = os.path.abspath(dir_output)
        self.subdir_db_probes = "db_probes"
        self.subdir_db_reference = "db_reference"

        self.n_jobs = n_jobs

    @pipeline_step_basic(step_name="Create Database")
    def create_oligo_database(
        self,
        region_ids: list | None,
        oligo_length_min: int,
        oligo_length_max: int,
        split_region: int,
        files_fasta_oligo_database: list[str],
        min_oligos_per_gene: int,
    ) -> OligoDatabase:
        """
        Create an initial oligo database by generating sequences using a sliding window approach
        with region splitting and performing pre-filtering based on isoform consensus.

        This is the first step of target probe design. The method:
        1. Generates candidate oligo sequences from input FASTA files using a sliding window approach
           across the specified length range, with large regions split into smaller chunks for memory efficiency
        2. Creates an `OligoDatabase` and loads the generated sequences
        3. Calculates reverse complement sequences and isoform consensus properties
        4. Pre-filters oligos based on isoform consensus threshold
        5. Removes regions with insufficient oligos after filtering

        The database stores sequences with sequence types "target" (original sequence), "oligo" (reverse
        complement), and "oligo_short" (shortened sequences for read length bias filtering).

        :param region_ids: List of gene identifiers (e.g., gene IDs) to target for probe design. If None,
            all genes present in the input FASTA files will be used.
        :type region_ids: list[str] | None
        :param oligo_length_min: Minimum length (in nucleotides) for target probe sequences.
        :type oligo_length_min: int
        :param oligo_length_max: Maximum length (in nucleotides) for target probe sequences.
        :type oligo_length_max: int
        :param split_region: Size of regions (in nucleotides) to split large genomic regions into when
            generating probe candidates. This helps manage memory usage for very long sequences.
        :type split_region: int
        :param files_fasta_oligo_database: List of paths to FASTA files containing sequences from which
            target probes will be generated. These files should contain genomic regions of interest
            (e.g., exons, exon-exon junctions).
        :type files_fasta_oligo_database: list[str]
        :param min_oligos_per_gene: Minimum number of oligos required per region (gene) after filtering.
            Regions with fewer oligos than this threshold will be removed from the database.
        :type min_oligos_per_gene: int
        :return: An `OligoDatabase` object containing the generated target probe sequences with their
            component sequences (target, oligo, oligo_short) and calculated properties (isoform_consensus).
            The database is filtered to only include regions that meet the minimum oligo requirement.
        :rtype: OligoDatabase
        """
        # generate candidate oligo sequences (sliding window over FASTA regions)
        oligo_sequences = OligoSequenceGenerator(dir_output=self.dir_output)
        oligo_fasta_file = oligo_sequences.create_sequences_sliding_window(
            files_fasta_in=files_fasta_oligo_database,
            length_interval_sequences=(oligo_length_min, oligo_length_max),
            split_region=split_region,
            region_ids=region_ids,
            n_jobs=self.n_jobs,
        )

        # load sequences into oligo database
        oligo_database = OligoDatabase(
            min_oligos_per_region=min_oligos_per_gene,
            write_regions_with_insufficient_oligos=True,
            max_entries_in_memory=self.n_jobs * 2 + 2,
            database_name=self.subdir_db_probes,
            dir_output=self.dir_output,
            n_jobs=1,
        )
        oligo_database.load_database_from_fasta(
            files_fasta=oligo_fasta_file,
            sequence_type="target",
            database_overwrite=True,
            region_ids=region_ids,
        )
        oligo_database.set_database_sequence_types(["target", "oligo", "oligo_short"])

        # compute reverse complement (oligo) and isoform consensus per entry
        rc_sequence_property = ReverseComplementSequenceProperty(sequence_type_reverse_complement="oligo")
        calculator = PropertyCalculator(properties=[rc_sequence_property])
        oligo_database = calculator.apply(
            oligo_database=oligo_database, sequence_type="target", n_jobs=self.n_jobs
        )

        # remove temporary sliding-window output directory
        dir = oligo_sequences.dir_output
        shutil.rmtree(dir) if os.path.exists(dir) else None

        # drop regions with too few oligos and validate database
        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Database Creation")
        check_content_oligo_database(oligo_database)

        return oligo_database

    @pipeline_step_basic(step_name="Property Filters")
    def filter_by_property(
        self,
        oligo_database: OligoDatabase,
        target_probe_isoform_consensus_filter: dict,
        target_probe_targeted_exons_filter: dict,
        target_probe_hard_masked_sequences_filter: dict,
        target_probe_soft_masked_sequences_filter: dict,
        target_probe_homopolymeric_runs_filter: dict,
        target_probe_GC_content_filter: dict,
        target_probe_prohibited_sequences_filter: dict,
        target_probe_self_complementarity_filter: dict,
        target_probe_melting_temperature_filter: dict,
        target_probe_secondary_structure_filter: dict,
    ) -> OligoDatabase:
        """
        Filter the oligo database based on sequence properties to remove probes with undesirable
        characteristics.

        This method applies sequential filtering using multiple property-based filters:
        1. **Hard masked sequences**: Removes probes containing hard-masked nucleotides (N)
        2. **Soft masked sequences**: Removes probes containing soft-masked nucleotides (lowercase)
        3. **Homopolymeric runs**: Removes probes with homopolymeric runs exceeding specified lengths
        4. **GC content**: Removes probes with GC content outside the specified range
        5. **Melting temperature**: Removes probes with Tm outside the specified range
        6. **Self-complementarity**: Removes probes with excessive self-complementary regions that can
           form hairpins and reduce hybridization efficiency
        7. **Secondary structure**: Removes probes that form stable secondary structures at the
           specified temperature

        Probes that fail any filter are removed. Regions with insufficient oligos after filtering
        are removed from the database.

        Each filter argument is a dict with an ``enabled`` key and filter-specific keys.

        :param oligo_database: The `OligoDatabase` instance containing oligonucleotide sequences
            and their associated properties. This database should contain target probes with their
            component sequences already calculated.
        :type oligo_database: OligoDatabase
        :param target_probe_isoform_consensus_filter: Dict with ``enabled``, ``isoform_consensus``.
        :type target_probe_isoform_consensus_filter: dict
        :param target_probe_targeted_exons_filter: Dict with ``enabled``, ``targeted_exons``.
        :type target_probe_targeted_exons_filter: dict
        :param target_probe_hard_masked_sequences_filter: Dict with ``enabled``.
        :type target_probe_hard_masked_sequences_filter: dict
        :param target_probe_soft_masked_sequences_filter: Dict with ``enabled``.
        :type target_probe_soft_masked_sequences_filter: dict
        :param target_probe_homopolymeric_runs_filter: Dict with ``enabled``, ``homopolymeric_base_n``.
        :type target_probe_homopolymeric_runs_filter: dict
        :param target_probe_GC_content_filter: Dict with ``enabled``, ``GC_content_min``, ``GC_content_max``.
        :type target_probe_GC_content_filter: dict
        :param target_probe_prohibited_sequences_filter: Dict with ``enabled``, ``prohibited_sequences``,
            optionally ``kmer_abundance_threshold``.
        :type target_probe_prohibited_sequences_filter: dict
        :param target_probe_self_complementarity_filter: Dict with ``enabled``, ``max_len_selfcomplement``.
        :type target_probe_self_complementarity_filter: dict
        :param target_probe_melting_temperature_filter: Dict with ``enabled``, ``Tm_min``, ``Tm_max``,
            ``Tm_parameters``, ``Tm_chem_correction_parameters``, ``Tm_salt_correction_parameters``.
        :type target_probe_melting_temperature_filter: dict
        :param target_probe_secondary_structure_filter: Dict with ``enabled``, ``T``, ``thr_DG``.
        :type target_probe_secondary_structure_filter: dict
        :return: A filtered `OligoDatabase` object containing only probes that pass all property filters.
            Regions with insufficient oligos after filtering are removed.
        :rtype: OligoDatabase
        """

        # Pre-filter by isoform consensus (cheap property lookup before sequence filters)
        if target_probe_isoform_consensus_filter["enabled"]:
            isoform_consensus_property = IsoformConsensusProperty()
            calculator = PropertyCalculator(properties=[isoform_consensus_property])
            oligo_database = calculator.apply(
                oligo_database=oligo_database, sequence_type="target", n_jobs=self.n_jobs
            )
            oligo_database.filter_database_by_property_threshold(
                property_name="isoform_consensus",
                property_thr=target_probe_isoform_consensus_filter["isoform_consensus"],
                remove_if_smaller_threshold=True,
            )

        if target_probe_targeted_exons_filter["enabled"]:
            targeted_exons_property = TargetedExonsProperty()
            calculator = PropertyCalculator(properties=[targeted_exons_property])
            oligo_database = calculator.apply(
                oligo_database=oligo_database, sequence_type="target", n_jobs=self.n_jobs
            )
            oligo_database.filter_database_by_property_category(
                property_name="targeted_exons",
                property_category=target_probe_targeted_exons_filter["targeted_exons"],
                remove_if_equals_category=False,
            )

        # Instantiate sequence-based property filters
        # Masking: drop oligos with ambiguous or low-complexity bases
        filters: list[BasePropertyFilter] = []
        if target_probe_hard_masked_sequences_filter["enabled"]:
            hard_masked_sequences = HardMaskedSequenceFilter()
            filters.append(hard_masked_sequences)

        if target_probe_soft_masked_sequences_filter["enabled"]:
            soft_masked_sequences = SoftMaskedSequenceFilter()
            filters.append(soft_masked_sequences)

        # Composition: homopolymeric runs, GC range, prohibited motifs
        if target_probe_homopolymeric_runs_filter["enabled"]:
            homopolymeric_runs = HomopolymericRunsFilter(
                base_n=target_probe_homopolymeric_runs_filter["homopolymeric_base_n"],
            )
            filters.append(homopolymeric_runs)

        if target_probe_GC_content_filter["enabled"]:
            gc_content = GCContentFilter(
                GC_content_min=target_probe_GC_content_filter["GC_content_min"],
                GC_content_max=target_probe_GC_content_filter["GC_content_max"],
            )
            filters.append(gc_content)

        if target_probe_prohibited_sequences_filter["enabled"]:
            prohibited_sequence_filter = ProhibitedSequenceFilter(
                prohibited_sequences=target_probe_prohibited_sequences_filter["prohibited_sequences"],
            )
            filters.append(prohibited_sequence_filter)

        # Thermodynamics: self-complementarity (hairpins), Tm range, secondary structure (ΔG)
        if target_probe_self_complementarity_filter["enabled"]:
            self_comp = SelfComplementFilter(
                max_len_selfcomplement=target_probe_self_complementarity_filter["max_len_selfcomplement"],
            )
            filters.append(self_comp)

        if target_probe_melting_temperature_filter["enabled"]:
            melting_temperature = MeltingTemperatureNNFilter(
                Tm_min=target_probe_melting_temperature_filter["Tm_min"],
                Tm_max=target_probe_melting_temperature_filter["Tm_max"],
                Tm_parameters=target_probe_melting_temperature_filter["Tm_parameters"],
                Tm_chem_correction_parameters=target_probe_melting_temperature_filter[
                    "Tm_chem_correction_parameters"
                ],
                Tm_salt_correction_parameters=target_probe_melting_temperature_filter[
                    "Tm_salt_correction_parameters"
                ],
            )
            filters.append(melting_temperature)

        if target_probe_secondary_structure_filter["enabled"]:
            secondary_structure = SecondaryStructureFilter(
                T=target_probe_secondary_structure_filter["T"],
                thr_DG=target_probe_secondary_structure_filter["thr_DG"],
            )
            filters.append(secondary_structure)

        # Apply filters in order of cost (cheapest first) so failing oligos are rejected early.
        property_filter = PropertyFilter(filters=filters)
        oligo_database = property_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo",
            n_jobs=self.n_jobs,
        )
        check_content_oligo_database(oligo_database)

        return oligo_database

    @pipeline_step_basic(step_name="Specificity Filters")
    def filter_by_specificity(
        self,
        oligo_database: OligoDatabase,
        target_probe_read_length_bias_filter: dict,
        target_probe_cross_hybridization_filter: dict,
        target_probe_specificity_blastn_filter: dict,
        target_probe_variant_filter: dict,
    ) -> OligoDatabase:
        """
        Filter the oligo database based on sequence specificity to remove probes that bind
        non-specifically, cross-hybridize, overlap with variants, or have high hybridization probability.

        This method applies multiple types of specificity filters:

        1. **Read length bias filtering**: Removes probes where the first N bases match exactly with
           other probes. This prevents sequencing read length biases that could occur when multiple
           probes share identical 5' ends.

        2. **Exact match filtering**: Removes all probes with exact sequence matches to probes of
           other regions.

        3. **Variant filtering**: Marks probes that overlap with known single nucleotide polymorphisms
           (SNPs) or other variants from VCF files. Probes overlapping variants may have reduced
           specificity and are flagged (not removed) for downstream analysis.

        4. **Cross-hybridization filtering**: Removes probes that cross-hybridize with each other.
           This is critical because if probes can bind to each other, they may form dimers instead
           of binding to the target RNA. Probes from the larger genomic region are removed when
           cross-hybridization is detected. The alignment method (BLASTN or Bowtie) can be selected.

        5. **Hybridization probability filtering**: Removes probes with high hybridization probability
           to unintended targets in the reference database. This filter uses alignment-based methods
           (BLASTN or Bowtie) to estimate the probability of non-specific binding and removes probes
           above the specified threshold.

        The reference databases are loaded from the provided FASTA and VCF files. Regions that do not
        meet the minimum oligo requirement after filtering are removed from the database.

        Each filter argument is a dict with ``enabled`` and filter-specific keys.

        :param oligo_database: The `OligoDatabase` instance containing oligonucleotide sequences
            and their associated properties. This database should contain target probes with their
            component sequences already calculated.
        :type oligo_database: OligoDatabase
        :param target_probe_read_length_bias_filter: Dict with ``enabled``, ``read_length_bias``.
        :type target_probe_read_length_bias_filter: dict
        :param target_probe_cross_hybridization_filter: Dict with ``enabled``,
            ``cross_hybridization_search_parameters``, ``cross_hybridization_hit_parameters``.
        :type target_probe_cross_hybridization_filter: dict
        :param target_probe_specificity_blastn_filter: Dict with ``enabled``,
            ``specificity_blastn_search_parameters``, ``specificity_blastn_hit_parameters``,
            ``files_fasta_reference_database``.
        :type target_probe_specificity_blastn_filter: dict
        :param target_probe_variant_filter: Dict with ``enabled``, ``files_vcf_reference_database``,
            ``action`` ("flag" or "remove").
        :type target_probe_variant_filter: dict
        :return: A filtered `OligoDatabase` object containing only probes that pass all specificity
            filters. Probes overlapping variants are flagged (or removed if action is "remove"). Regions
            with insufficient oligos after filtering are removed.
        :rtype: OligoDatabase
        """
        # remove sequences that could cause read length biases because the first
        # <target_probe_read_length_bias> bases of both sequences match
        # Calculate shortened sequence using new PropertyCalculator pattern
        if target_probe_read_length_bias_filter["enabled"]:
            shortened_sequence_property = ShortenedSequenceProperty(
                sequence_length=target_probe_read_length_bias_filter["read_length_bias"], reverse=False
            )
            calculator = PropertyCalculator(properties=[shortened_sequence_property])
            oligo_database = calculator.apply(
                oligo_database=oligo_database, sequence_type="oligo", n_jobs=self.n_jobs
            )
            exact_matches_short = ExactMatchFilter(
                policy=RemoveAllFilterPolicy(), filter_name="exact_match_read_length_bias"
            )
            specificity_filter = SpecificityFilter(filters=[exact_matches_short])
            oligo_database = specificity_filter.apply(
                oligo_database=oligo_database,
                sequence_type="oligo_short",
                n_jobs=self.n_jobs,
            )

        exact_matches = ExactMatchFilter(policy=RemoveAllFilterPolicy(), filter_name="exact_match")
        filters: list[BaseSpecificityFilter] = [exact_matches]
        directories = []

        if target_probe_cross_hybridization_filter["enabled"]:
            cross_hybridization_aligner = BlastNFilter(
                remove_hits=True,
                search_parameters=target_probe_cross_hybridization_filter[
                    "cross_hybridization_search_parameters"
                ],
                hit_parameters=target_probe_cross_hybridization_filter["cross_hybridization_hit_parameters"],
                filter_name="cross_hybridization_filter",
                dir_output=self.dir_output,
            )
            cross_hybridization = CrossHybridizationFilter(
                policy=RemoveByLargerRegionFilterPolicy(),
                alignment_method=cross_hybridization_aligner,
                filter_name="cross_hybridization_filter",
                dir_output=self.dir_output,
            )
            filters.append(cross_hybridization)
            directories.append(cross_hybridization_aligner.dir_output)
            directories.append(cross_hybridization.dir_output)

        if target_probe_specificity_blastn_filter["enabled"]:
            reference_database_alignment = ReferenceDatabase(
                database_name=f"{self.subdir_db_reference}_sequences", dir_output=self.dir_output
            )
            reference_database_alignment.load_database_from_file(
                files=target_probe_specificity_blastn_filter["files_fasta_reference_database"],
                file_type="fasta",
                database_overwrite=True,
            )
            specificity = BlastNFilter(
                remove_hits=True,
                search_parameters=target_probe_specificity_blastn_filter[
                    "specificity_blastn_search_parameters"
                ],
                hit_parameters=target_probe_specificity_blastn_filter["specificity_blastn_hit_parameters"],
                filter_name="oligo_blastn_specificity",
                dir_output=self.dir_output,
            )
            specificity.set_reference_database(reference_database=reference_database_alignment)
            filters.append(specificity)
            directories.append(specificity.dir_output)

        if target_probe_variant_filter["enabled"]:
            remove_hits = target_probe_variant_filter["action"] == "remove"
            reference_database_variants = ReferenceDatabase(
                database_name=f"{self.subdir_db_reference}_variants", dir_output=self.dir_output
            )
            reference_database_variants.load_database_from_file(
                files=target_probe_variant_filter["files_vcf_reference_database"],
                file_type="vcf",
                database_overwrite=True,
            )
            variants = VariantsFilter(
                remove_hits=remove_hits, filter_name="SNP_filter", dir_output=self.dir_output
            )
            variants.set_reference_database(reference_database=reference_database_variants)
            filters.append(variants)
            directories.append(variants.dir_output)

        # run all filters specified above
        specificity_filter = SpecificityFilter(filters=filters)
        oligo_database = specificity_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo",
            n_jobs=self.n_jobs,
        )

        # remove all directories of intermediate steps
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)

        check_content_oligo_database(oligo_database)

        return oligo_database

    @pipeline_step_basic(step_name="Oligo Selection")
    def create_oligo_sets(
        self,
        oligo_database: OligoDatabase,
        target_probe_set_selection_parameters: dict,
    ) -> OligoDatabase:
        """
        Create optimal oligo sets based on weighted scoring criteria, distance constraints, and set selection.

        This method performs the following steps:
        1. **Scoring**: Calculates scores for each oligo based on weighted criteria (targeted exons overlap,
           isoform consensus, GC content, melting temperature). Higher scores indicate better probes.
        2. **Set generation**: Builds a compatibility graph from distance constraints and selects sets via
           a graph-based (clique) strategy. Generates multiple diverse sets per region, controlling overlap
           between sets using a Jaccard threshold (`jaccard_opt`) with optional relaxation (`jaccard_step`).
        3. **Set scoring**: Evaluates each generated set by average oligo score and selects the best sets
           (ascending order, i.e. lower average score is better for this pipeline).
        4. **Region filtering**: Removes regions that cannot generate sets meeting the minimum size requirement.

        The algorithm attempts to find sets with optimal size (`set_size_opt`) but may produce sets
        as small as `set_size_min` if constraints cannot be met.

        :param oligo_database: The `OligoDatabase` instance containing oligonucleotide sequences
            and their associated properties. This database should contain target probes that have
            passed all previous filtering steps.
        :type oligo_database: OligoDatabase
        :param target_probe_set_selection_parameters: Set selection config. Keys: ``n_sets``, ``set_size_min``,
            ``set_size_opt``, ``distance_between_target_probes``, ``uniform_distance_weight``, ``isoform_weight``,
            ``targeted_exons_weight``, ``targeted_exons``, ``GC_weight``, ``GC_content_opt``, ``Tm_weight``, ``Tm_opt``,
            ``n_attempts_graph``, ``n_attempts_clique_enum``, ``diversification_fraction``, ``jaccard_opt``,
            ``jaccard_step``, ``Tm_parameters``, ``Tm_chem_correction_parameters``,
            ``Tm_salt_correction_parameters``, ``GC_content_min``, ``GC_content_max``, ``Tm_min``, ``Tm_max``.
        :type target_probe_set_selection_parameters: dict
        :return: An updated `OligoDatabase` object containing the generated oligo sets. Each region
            will have up to `n_sets` sets stored, with each set containing between `set_size_min` and
            `set_size_opt` probes. Regions with insufficient oligos are removed.
        :rtype: OligoDatabase
        """
        oligo_lengths = [
            len(sequence) for sequence in oligo_database.get_sequence_list(sequence_type="oligo")
        ]
        average_oligo_length = sum(oligo_lengths) / len(oligo_lengths)

        # Define all scorers
        uniform_distance_scorer = UniformDistanceScorer(
            average_oligo_length=average_oligo_length,
            score_weight=target_probe_set_selection_parameters["uniform_distance_weight"],
        )
        exon_scorer = OverlapTargetedExonsScorer(
            targeted_exons=target_probe_set_selection_parameters["targeted_exons"],
            score_weight=target_probe_set_selection_parameters["targeted_exons_weight"],
            property_name="exon_number",
        )
        isoform_scorer = IsoformConsensusScorer(
            score_weight=target_probe_set_selection_parameters["isoform_weight"]
        )
        Tm_scorer = NormalizedDeviationFromOptimalTmScorer(
            Tm_min=target_probe_set_selection_parameters["Tm_min"],
            Tm_opt=target_probe_set_selection_parameters["Tm_opt"],
            Tm_max=target_probe_set_selection_parameters["Tm_max"],
            Tm_parameters=target_probe_set_selection_parameters["Tm_parameters"],
            Tm_chem_correction_parameters=target_probe_set_selection_parameters[
                "Tm_chem_correction_parameters"
            ],
            Tm_salt_correction_parameters=target_probe_set_selection_parameters[
                "Tm_salt_correction_parameters"
            ],
            score_weight=target_probe_set_selection_parameters["Tm_weight"],
        )
        GC_scorer = NormalizedDeviationFromOptimalGCContentScorer(
            GC_content_min=target_probe_set_selection_parameters["GC_content_min"],
            GC_content_opt=target_probe_set_selection_parameters["GC_content_opt"],
            GC_content_max=target_probe_set_selection_parameters["GC_content_max"],
            score_weight=target_probe_set_selection_parameters["GC_weight"],
        )

        oligos_scoring = OligoScoring(
            scorers=[exon_scorer, isoform_scorer, Tm_scorer, GC_scorer, uniform_distance_scorer]
        )
        set_scoring = AverageSetScoring(ascending=True)

        base_log_parameters({"Set Selection": "Independent Sets"})
        oligoset_generator = IndependentSetsOligoSelection(
            oligos_scoring=oligos_scoring,
            set_scoring=set_scoring,
            set_size_opt=target_probe_set_selection_parameters["set_size_opt"],
            set_size_min=target_probe_set_selection_parameters["set_size_min"],
            distance_between_oligos=target_probe_set_selection_parameters["distance_between_target_probes"],
            n_attempts_graph=target_probe_set_selection_parameters["n_attempts_graph"],
            n_attempts_clique_enum=target_probe_set_selection_parameters["n_attempts_clique_enum"],
            diversification_fraction=target_probe_set_selection_parameters["diversification_fraction"],
            jaccard_opt=target_probe_set_selection_parameters["jaccard_opt"],
            jaccard_step=target_probe_set_selection_parameters["jaccard_step"],
        )
        oligo_database = oligoset_generator.apply(
            oligo_database=oligo_database,
            sequence_type="oligo",
            n_sets=target_probe_set_selection_parameters["n_sets"],
            n_jobs=self.n_jobs,
        )
        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Oligo Selection")
        check_content_oligo_database(oligo_database)

        return oligo_database


############################################
# Oligo-seq Designer Pipeline
############################################


def _parse_config(config_file: str) -> dict[str, Any]:
    """
    Load config from YAML, validate required and filter parameters, return config in same dict structure.

    - Required (non-filter) top-level keys must be present or ValueError is raised.
    - If a filter block is present and enabled, its required parameters must be set or ValueError is raised.
    - Returns the config dict in the same nested format as the YAML (e.g. filter blocks like
      target_probe_isoform_consensus_filter: { enabled: True, isoform_consensus: 0 }).

    :param config_file: Path to the YAML configuration file.
    :type config_file: str
    :return: Validated configuration dictionary with the same nested structure as the YAML file.
    :rtype: dict[str, Any]
    :raises ValueError: If a required top-level key, design parameter, set-selection parameter, or
        enabled filter parameter is missing.
    """
    # -----  Required keys (cannot be disabled / always needed)  -----
    required_top: list[str] = [
        "n_jobs",
        "dir_output",
        "write_intermediate_steps",
        "target_probe_design_parameters",
        "target_probe_set_selection_parameters",
        "target_probe_Tm_parameters",
        "target_probe_Tm_chem_correction_parameters",
        "target_probe_Tm_salt_correction_parameters",
    ]
    required_design: list[str] = [
        "file_region_ids",
        "files_fasta_probe_database",
        "probe_length_min",
        "probe_length_max",
        "probe_split_region",
    ]
    filter_required_params: dict[str, list[str]] = {
        "target_probe_isoform_consensus_filter": ["isoform_consensus"],
        "target_probe_hard_masked_sequences_filter": [],
        "target_probe_soft_masked_sequences_filter": [],
        "target_probe_homopolymeric_runs_filter": ["homopolymeric_base_n"],
        "target_probe_GC_content_filter": ["GC_content_min", "GC_content_max"],
        "target_probe_prohibited_sequences_filter": ["prohibited_sequences"],
        "target_probe_self_complementarity_filter": ["max_len_selfcomplement"],
        "target_probe_melting_temperature_filter": ["Tm_min", "Tm_max"],
        "target_probe_secondary_structure_filter": ["T", "thr_DG"],
        "target_probe_read_length_bias_filter": ["read_length_bias"],
        "target_probe_cross_hybridization_filter": [
            "cross_hybridization_search_parameters",
            "cross_hybridization_hit_parameters",
        ],
        "target_probe_specificity_blastn_filter": [
            "specificity_blastn_search_parameters",
            "specificity_blastn_hit_parameters",
            "files_fasta_reference_database",
        ],
        "target_probe_variant_filter": ["files_vcf_reference_database"],
    }
    required_selection: list[str] = [
        "n_sets",
        "set_size_min",
        "set_size_opt",
        "distance_between_target_probes",
        "uniform_distance_weight",
        "isoform_weight",
        "targeted_exons_weight",
        "targeted_exons",
        "GC_weight",
        "GC_content_opt",
        "Tm_weight",
        "Tm_opt",
        "n_attempts_graph",
        "n_attempts_clique_enum",
        "diversification_fraction",
        "jaccard_opt",
        "jaccard_step",
    ]

    with open(config_file, "r") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required parameter: '{key}'.")

    # -----  Required keys inside target_probe_design_parameters  -----
    design = config["target_probe_design_parameters"]
    if not isinstance(design, dict):
        raise ValueError("'target_probe_design_parameters' must be a mapping.")

    for key in required_design:
        if key not in design:
            raise ValueError(f"Missing required parameter in target_probe_design_parameters: '{key}'.")

    # -----  Required keys inside target_probe_set_selection_parameters  -----
    selection = config["target_probe_set_selection_parameters"]
    if not isinstance(selection, dict):
        raise ValueError("'target_probe_set_selection_parameters' must be a mapping.")

    for key in required_selection:
        if key not in selection:
            raise ValueError(f"Missing required parameter in target_probe_set_selection_parameters: '{key}'.")

    # -----  Filter blocks: when enabled, required parameters must be set  -----
    # Map filter key -> list of required param keys (besides "enabled")
    for filter_key, required_params in filter_required_params.items():
        block = config.get(filter_key)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"Filter block '{filter_key}' must be a mapping.")
        enabled = block.get("enabled")
        if not enabled:
            continue
        for param in required_params:
            if param not in block:
                raise ValueError(
                    f"Filter '{filter_key}' is enabled but missing required parameter: '{param}'."
                )

    return config


def _preprocess_config(config: dict) -> dict:
    """
    Preprocess the pipeline config: format Tm parameters and inject them into filter and set-selection blocks.

    - Preprocesses Tm parameters (nn_table, salt, etc.) and injects them into
      target_probe_melting_temperature_filter and target_probe_set_selection_parameters.
    - Injects GC_content_min/max and Tm_min/max from the respective filters into
      target_probe_set_selection_parameters for scoring.
    - Reads the region IDs file (if provided) and sets target_probe_design_parameters["region_ids"].

    :param config: Validated configuration dictionary (e.g. from _parse_config). Modified in place and returned.
    :type config: dict
    :return: The same config dict after preprocessing (modified in place).
    :rtype: dict
    """
    config["target_probe_Tm_parameters"] = preprocess_tm_parameters(config["target_probe_Tm_parameters"])

    if config["target_probe_melting_temperature_filter"]["enabled"]:
        config["target_probe_melting_temperature_filter"]["Tm_parameters"] = config[
            "target_probe_Tm_parameters"
        ]
        config["target_probe_melting_temperature_filter"]["Tm_chem_correction_parameters"] = config[
            "target_probe_Tm_chem_correction_parameters"
        ]
        config["target_probe_melting_temperature_filter"]["Tm_salt_correction_parameters"] = config[
            "target_probe_Tm_salt_correction_parameters"
        ]

    config["target_probe_set_selection_parameters"]["Tm_parameters"] = config["target_probe_Tm_parameters"]
    config["target_probe_set_selection_parameters"]["Tm_chem_correction_parameters"] = config[
        "target_probe_Tm_chem_correction_parameters"
    ]
    config["target_probe_set_selection_parameters"]["Tm_salt_correction_parameters"] = config[
        "target_probe_Tm_salt_correction_parameters"
    ]

    config["target_probe_set_selection_parameters"]["GC_content_min"] = config[
        "target_probe_GC_content_filter"
    ]["GC_content_min"]
    config["target_probe_set_selection_parameters"]["GC_content_max"] = config[
        "target_probe_GC_content_filter"
    ]["GC_content_max"]
    config["target_probe_set_selection_parameters"]["Tm_min"] = config[
        "target_probe_melting_temperature_filter"
    ]["Tm_min"]
    config["target_probe_set_selection_parameters"]["Tm_max"] = config[
        "target_probe_melting_temperature_filter"
    ]["Tm_max"]

    ##### read the genes file #####
    file_region_ids = config["target_probe_design_parameters"]["file_region_ids"]
    if file_region_ids is None:
        warnings.warn(
            "No gene list file was provided! All genes from fasta file are used to generate the probes. This chioce can use a lot of resources."
        )
        config["target_probe_design_parameters"]["region_ids"] = None
    else:
        with open(file_region_ids) as f:
            config["target_probe_design_parameters"]["region_ids"] = sorted({line.rstrip() for line in f})

    return config


def main() -> None:
    """
    Main entry point for running the Oligo-seq probe design pipeline.

    This function orchestrates the complete Oligo-seq probe design workflow:
    1. Parses command-line arguments using the base parser
    2. Reads the configuration YAML file containing all pipeline parameters
    3. Reads the gene IDs file (if provided) or uses all genes from FASTA files
    4. Preprocesses melting temperature parameters for target probes
    5. Preprocesses alignment method parameters for hybridization probability and cross-hybridization
       filtering (BLASTN or Bowtie)
    6. Initializes the OligoSeqProbeDesigner pipeline
    7. Designs target probes for specified genes
    8. Generates output files (YAML, TSV, Excel, order file)

    The function is typically called from the command line:
    ``oligo_seq_probe_designer --config <path_to_config.yaml>``

    Command-line arguments are parsed using `base_parser()`, which expects:
    - `config`: Path to the YAML configuration file containing all pipeline parameters.

    :return: None
    """
    logging.info("--------------START PIPELINE--------------")

    args = base_parser()
    config = _parse_config(args["config"])
    config = _preprocess_config(config)

    ##### initialize probe designer pipeline #####
    pipeline = OligoSeqProbeDesigner(
        write_intermediate_steps=config["write_intermediate_steps"],
        dir_output=config["dir_output"],
        n_jobs=config["n_jobs"],
    )

    ##### design probes #####
    oligo_database = pipeline.design_target_probes(
        # Step 1: Create Database Parameters
        target_probe_design_parameters=config["target_probe_design_parameters"],
        # Step 2: Property Filter Parameters
        target_probe_isoform_consensus_filter=config["target_probe_isoform_consensus_filter"],
        target_probe_targeted_exons_filter=config["target_probe_targeted_exons_filter"],
        target_probe_hard_masked_sequences_filter=config["target_probe_hard_masked_sequences_filter"],
        target_probe_soft_masked_sequences_filter=config["target_probe_soft_masked_sequences_filter"],
        target_probe_homopolymeric_runs_filter=config["target_probe_homopolymeric_runs_filter"],
        target_probe_GC_content_filter=config["target_probe_GC_content_filter"],
        target_probe_prohibited_sequences_filter=config["target_probe_prohibited_sequences_filter"],
        target_probe_self_complementarity_filter=config["target_probe_self_complementarity_filter"],
        target_probe_melting_temperature_filter=config["target_probe_melting_temperature_filter"],
        target_probe_secondary_structure_filter=config["target_probe_secondary_structure_filter"],
        # Step 3: Specificity Filter Parameters
        target_probe_read_length_bias_filter=config["target_probe_read_length_bias_filter"],
        target_probe_cross_hybridization_filter=config["target_probe_cross_hybridization_filter"],
        target_probe_specificity_blastn_filter=config["target_probe_specificity_blastn_filter"],
        target_probe_variant_filter=config["target_probe_variant_filter"],
        # Step 4: Probe Scoring and Set Selection Parameters
        target_probe_set_selection_parameters=config["target_probe_set_selection_parameters"],
    )

    pipeline.generate_output(oligo_database=oligo_database)

    logging.info("--------------END PIPELINE--------------")


if __name__ == "__main__":
    main()
