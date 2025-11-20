############################################
# imports
############################################

import itertools
import logging
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import Seq

from oligo_designer_toolsuite._exceptions import (
    ConfigurationError,
    FeatureNotImplementedError,
    FileFormatError,
)
from oligo_designer_toolsuite.database import OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_efficiency_filter import (
    AverageSetScoring,
    DeviationFromOptimalTmScorer,
    IsoformConsensusScorer,
    OligoScoring,
)
from oligo_designer_toolsuite.oligo_property_calculator import (
    BaseProperty,
    IsoformConsensusProperty,
    PropertyCalculator,
    ReverseComplementSequenceProperty,
    SplitSequenceProperty,
    TmNNProperty,
)
from oligo_designer_toolsuite.oligo_property_filter import (
    GCContentFilter,
    HardMaskedSequenceFilter,
    HomopolymericRunsFilter,
    MeltingTemperatureNNFilter,
    PropertyFilter,
    SecondaryStructureFilter,
)
from oligo_designer_toolsuite.oligo_selection import (
    GraphBasedSelectionPolicy,
    GreedySelectionPolicy,
    OligoSelectionPolicy,
    OligosetGeneratorIndependentSet,
)
from oligo_designer_toolsuite.oligo_specificity_filter import (
    AlignmentSpecificityFilter,
    BlastNFilter,
    BlastNSeedregionSiteFilter,
    CrossHybridizationFilter,
    ExactMatchFilter,
    RemoveAllFilterPolicy,
    RemoveByLargerRegionFilterPolicy,
    SpecificityFilter,
)
from oligo_designer_toolsuite.pipelines._utils import (
    base_log_parameters,
    base_parser,
    check_content_oligo_database,
    format_sequence,
    pipeline_step_basic,
    setup_logging,
)
from oligo_designer_toolsuite.sequence_generator import OligoSequenceGenerator

############################################
# CycleHCR Probe Designer
############################################


class CycleHCRProbeDesigner:
    """
    A class for designing encoding probes for the CycleHCR experiments.

    A CycleHCR encoding probe is a fluorescent probe that contains a 92-nt targeting sequence (divided into
    45-nt segments for the left and right probe pairs, separated by a 2-nt gap), which directs their binding
    to the specific RNA, two 14-nt barcode sequences, which are read out by fluorescent secondary readout probes,
    TT-nucleotide spacers between readout and gene-specific regions, and two PCR primer binding sites.
    The specific readout sequences contained by an encoding probe are determined by the binary barcode assigned to that RNA.

    :param write_intermediate_steps: Whether to save intermediate results during the probe design pipeline.
    :type write_intermediate_steps: bool
    :param dir_output: Directory path where output files will be saved.
    :type dir_output: str
    :param n_jobs: Number of parallel jobs to use for processing.
    :type n_jobs: int
    """

    def __init__(
        self,
        write_intermediate_steps: bool,
        dir_output: str,
        n_jobs: int,
        output_properties: list[str] | None = None,
    ) -> None:
        """Constructor for the CycleHCRProbeDesigner class."""

        # create the output folder
        self.dir_output = os.path.abspath(dir_output)
        Path(self.dir_output).mkdir(parents=True, exist_ok=True)

        # setup logger
        setup_logging(
            dir_output=self.dir_output,
            pipeline_name="cyclehcr_probe_designer",
            log_start_message=True,
        )

        ##### set class parameters #####
        self.write_intermediate_steps = write_intermediate_steps
        self.n_jobs = n_jobs
        self.set_developer_parameters()

        ##### define output properties #####
        if output_properties is None:
            self.output_properties = [
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
                "sequence_target",
                "sequence_spacer",
                "sequence_readout_probe_L",
                "sequence_readout_probe_R",
                "sequence_hybridization_probe_L",
                "sequence_hybridization_probe_R",
                "sequence_forward_primer",
                "sequence_reverse_primer",
                "sequence_dna_template_probe_L",
                "sequence_dna_template_probe_R",
                "TmNN_sequence_target_L",
                "TmNN_sequence_target_R",
                "isoform_consensus",
            ]
        else:
            self.output_properties = output_properties

    def set_developer_parameters(
        self,
        target_probe_specificity_blastn_search_parameters: dict = {
            "-perc_identity": 80,
            "-strand": "plus",
            "-word_size": 10,
            "-dust": "no",
            "-soft_masking": "false",
            "-max_target_seqs": 10,
            "-max_hsps": 1000,
        },
        target_probe_specificity_blastn_hit_parameters: dict = {"coverage": 50},
        target_probe_cross_hybridization_blastn_search_parameters: dict = {
            "-perc_identity": 80,
            "-strand": "minus",
            "-word_size": 7,
            "-dust": "no",
            "-soft_masking": "false",
            "-max_target_seqs": 10,
        },
        target_probe_cross_hybridization_blastn_hit_parameters: dict = {"coverage": 50},
        target_probe_Tm_parameters: dict = {
            "check": True,
            "strict": True,
            "c_seq": None,
            "shift": 0,
            "nn_table": "DNA_NN3",
            "tmm_table": "DNA_TMM1",
            "imm_table": "DNA_IMM1",
            "de_table": "DNA_DE1",
            "dnac1": 25,
            "dnac2": 25,
            "selfcomp": False,
            "saltcorr": 0,
            "Na": 50,
            "K": 0,
            "Tris": 0,
            "Mg": 0,
            "dNTPs": 0,
        },
        target_probe_Tm_chem_correction_parameters: dict | None = None,
        target_probe_Tm_salt_correction_parameters: dict | None = None,
        max_graph_size: int = 5000,
        n_attempts: int = 100000,
        heuristic: bool = True,
        heuristic_n_attempts: int = 100,
    ) -> None:
        """
        Set developer-specific parameters for CycleHCR probe designer pipeline.
        These parameters can be used to customize and fine-tune the pipeline.

        :param target_probe_specificity_blastn_search_parameters: Parameters for the BlastN specificity
            search for target probes.
        :type target_probe_specificity_blastn_search_parameters: dict, optional
        :param target_probe_specificity_blastn_hit_parameters: Parameters for filtering BlastN hits
            for target probe specificity.
        :type target_probe_specificity_blastn_hit_parameters: dict, optional
        :param target_probe_cross_hybridization_blastn_search_parameters: Parameters for the BlastN
            cross-hybridization search for target probes.
        :type target_probe_cross_hybridization_blastn_search_parameters: dict, optional
        :param target_probe_cross_hybridization_blastn_hit_parameters: Parameters for filtering
            BlastN hits for target probe cross-hybridization.
        :type target_probe_cross_hybridization_blastn_hit_parameters: dict, optional
        :param target_probe_Tm_parameters: Parameters for calculating melting temperature (Tm) of target probes.
            For using Bio.SeqUtils.MeltingTemp default parameters set to ``{}``. For more information on parameters,
            see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.Tm_NN
        :type target_probe_Tm_parameters: dict
        :param target_probe_Tm_chem_correction_parameters: Chemical correction parameters for Tm calculation of target probes.
            For using Bio.SeqUtils.MeltingTemp default parameters set to ``{}``. For more information on parameters,
            see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.salt_correction
        :type target_probe_Tm_chem_correction_parameters: dict
        :param target_probe_Tm_salt_correction_parameters: Salt correction parameters for Tm calculation of target probes.
            For using Bio.SeqUtils.MeltingTemp default parameters set to ``{}``. For more information on parameters,
            see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.chem_correction
        :type target_probe_Tm_salt_correction_parameters: dict
        :param max_graph_size: Maximum size of the graph used in set selection, defaults to 5000.
        :type max_graph_size: int
        :param n_attempts: Maximum number of attempts for selecting oligo sets, defaults to 100000.
        :type n_attempts: int
        :param heuristic: Whether to apply heuristic methods in oligo set selection, defaults to True.
        :type heuristic: bool
        :param heuristic_n_attempts: Maximum number of attempts for heuristic selecting oligo sets, defaults to 100.
        :type heuristic_n_attempts: int
        """
        ### Parameters for the specificity filters
        # Specificity filter with BlastN
        self.target_probe_specificity_blastn_search_parameters = (
            target_probe_specificity_blastn_search_parameters
        )
        self.target_probe_specificity_blastn_hit_parameters = target_probe_specificity_blastn_hit_parameters

        # Crosshybridization filter with BlastN
        self.target_probe_cross_hybridization_blastn_search_parameters = (
            target_probe_cross_hybridization_blastn_search_parameters
        )
        self.target_probe_cross_hybridization_blastn_hit_parameters = (
            target_probe_cross_hybridization_blastn_hit_parameters
        )

        ### Parameters for Melting Temperature
        # The melting temperature is used in 2 different stages (property filters and padlock detection probe design), where a few parameters are shared and the others differ.
        # parameters for melting temperature -> for more information on parameters, see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.Tm_NN

        # preprocess melting temperature params
        target_probe_Tm_parameters["nn_table"] = getattr(mt, target_probe_Tm_parameters["nn_table"])
        target_probe_Tm_parameters["tmm_table"] = getattr(mt, target_probe_Tm_parameters["tmm_table"])
        target_probe_Tm_parameters["imm_table"] = getattr(mt, target_probe_Tm_parameters["imm_table"])
        target_probe_Tm_parameters["de_table"] = getattr(mt, target_probe_Tm_parameters["de_table"])

        ## target probe
        self.target_probe_Tm_parameters = target_probe_Tm_parameters
        self.target_probe_Tm_chem_correction_parameters = target_probe_Tm_chem_correction_parameters
        self.target_probe_Tm_salt_correction_parameters = target_probe_Tm_salt_correction_parameters

        ### Parameters for the Oligo set selection
        self.max_graph_size = max_graph_size
        self.heuristic = heuristic
        self.n_attempts = n_attempts
        self.heuristic_n_attempts = heuristic_n_attempts

    def design_target_probes(
        self,
        files_fasta_target_probe_database: list[str],
        files_fasta_reference_database_target_probe: list[str],
        region_ids: list[str] | None = None,
        target_probe_isoform_consensus: float = 0,
        target_probe_L_probe_sequence_length: int = 45,
        target_probe_gap_sequence_length: int = 2,
        target_probe_R_probe_sequence_length: int = 45,
        target_probe_GC_content_min: float = 43,
        target_probe_GC_content_max: float = 63,
        target_probe_Tm_min: float = 66,
        target_probe_Tm_max: float = 76,
        target_probe_homopolymeric_base_n: dict = {"A": 5, "T": 5, "C": 5, "G": 5},
        target_probe_T_secondary_structure: float = 76,
        target_probe_secondary_structures_threshold_deltaG: float = 0,
        target_probe_junction_region_size: int = 13,
        target_probe_Tm_weight: float = 1,
        target_probe_isoform_weight: float = 2,
        set_size_opt: int = 50,
        set_size_min: int = 50,
        distance_between_target_probes: int = 0,
        n_sets: int = 100,
    ) -> OligoDatabase:
        """
        Design target probes based on specified parameters, including property and specificity filters.
        The designed probes are organized into sets based on customizable constraints.

        :param files_fasta_target_probe_database: List of input FASTA files for the target probe database.
        :type files_fasta_target_probe_database: list[str]
        :param files_fasta_reference_database_target_probe: List of input FASTA files for the reference database.
        :type files_fasta_reference_database_target_probe: list[str]
        :param region_ids: List of region IDs to target, or None to target all regions.
        :type region_ids: list[str], optional
        :param target_probe_isoform_consensus: Isoform consensus threshold for filtering. Default is 50.
        :type target_probe_isoform_consensus: float
        :param target_probe_L_probe_sequence_length: Length of the left probe sequence. Default is 45.
        :type target_probe_L_probe_sequence_length: int
        :param target_probe_gap_sequence_length: Length of the gap sequence between left and right probes. Default is 2.
        :type target_probe_gap_sequence_length: int
        :param target_probe_R_probe_sequence_length: Length of the right probe sequence. Default is 45.
        :type target_probe_R_probe_sequence_length: int
        :param target_probe_GC_content_min: Minimum GC content for target probes. Default is 43.
        :type target_probe_GC_content_min: float
        :param target_probe_GC_content_max: Maximum GC content for target probes. Default is 63.
        :type target_probe_GC_content_max: float
        :param target_probe_Tm_min: Minimum melting temperature (Tm) for target probes. Default is 66.
        :type target_probe_Tm_min: float
        :param target_probe_Tm_max: Maximum melting temperature (Tm) for target probes. Default is 76.
        :type target_probe_Tm_max: float
        :param target_probe_homopolymeric_base_n: Maximum allowed homopolymeric runs for each nucleotide. Default is {"A": 5, "T": 5, "C": 5, "G": 5}.
        :type target_probe_homopolymeric_base_n: dict[str, int]
        :param target_probe_T_secondary_structure: Threshold temperature for secondary structure evaluation. Default is 76.
        :type target_probe_T_secondary_structure: float
        :param target_probe_secondary_structures_threshold_deltaG: DeltaG threshold for secondary structure stability. Default is 0.
        :type target_probe_secondary_structures_threshold_deltaG: float
        :param target_probe_junction_region_size: Size of the junction region for specificity filtering. Default is 13.
        :type target_probe_junction_region_size: int
        :param target_probe_Tm_weight: Weight for Tm in probe scoring. Default is 1.
        :type target_probe_Tm_weight: float
        :param target_probe_isoform_weight: Weight for isoform consensus in probe scoring. Default is 2.
        :type target_probe_isoform_weight: float
        :param set_size_opt: Optimal size of oligo sets. Default is 50.
        :type set_size_opt: int
        :param set_size_min: Minimum size of oligo sets. Default is 50.
        :type set_size_min: int
        :param distance_between_target_probes: Minimum genomic distance between probes in a set, defaults to 0.
        :type distance_between_target_probes: int, optional
        :param n_sets: Number of oligo sets to generate. Default is 100.
        :type n_sets: int
        :return: An `OligoDatabase` object containing the designed target probes.
        :rtype: OligoDatabase
        """

        target_probe_designer = TargetProbeDesigner(self.dir_output, self.n_jobs)

        oligo_database: OligoDatabase = target_probe_designer.create_oligo_database(
            region_ids=region_ids,
            target_probe_L_probe_sequence_length=target_probe_L_probe_sequence_length,
            target_probe_gap_sequence_length=target_probe_gap_sequence_length,
            target_probe_R_probe_sequence_length=target_probe_R_probe_sequence_length,
            files_fasta_oligo_database=files_fasta_target_probe_database,
            min_oligos_per_gene=set_size_min,
            isoform_consensus=target_probe_isoform_consensus,
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="1_db_target_probes_initial")
            logging.info(
                f"Saved target probe database for step 1 (Create Database) in directory {dir_database}"
            )

        oligo_database = target_probe_designer.filter_by_property(
            oligo_database=oligo_database,
            GC_content_min=target_probe_GC_content_min,
            GC_content_max=target_probe_GC_content_max,
            Tm_min=target_probe_Tm_min,
            Tm_max=target_probe_Tm_max,
            homopolymeric_base_n=target_probe_homopolymeric_base_n,
            T_secondary_structure=target_probe_T_secondary_structure,
            secondary_structures_threshold_deltaG=target_probe_secondary_structures_threshold_deltaG,
            Tm_parameters=self.target_probe_Tm_parameters,
            Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="2_db_target_probes_property_filter")
            logging.info(
                f"Saved target probe database for step 2 (Property Filters) in directory {dir_database}"
            )

        oligo_database = target_probe_designer.filter_by_specificity(
            oligo_database=oligo_database,
            files_fasta_reference_database=files_fasta_reference_database_target_probe,
            junction_region_size=target_probe_junction_region_size,
            junction_site=target_probe_L_probe_sequence_length + target_probe_gap_sequence_length // 2,
            specificity_blastn_search_parameters=self.target_probe_specificity_blastn_search_parameters,
            specificity_blastn_hit_parameters=self.target_probe_specificity_blastn_hit_parameters,
            cross_hybridization_blastn_search_parameters=self.target_probe_cross_hybridization_blastn_search_parameters,
            cross_hybridization_blastn_hit_parameters=self.target_probe_cross_hybridization_blastn_hit_parameters,
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="3_db_target_probes_specificity_filter")
            logging.info(
                f"Saved target probe database for step 3 (Specificity Filters) in directory {dir_database}"
            )

        oligo_database = target_probe_designer.create_oligo_sets(
            oligo_database=oligo_database,
            isoform_weight=target_probe_isoform_weight,
            Tm_max=target_probe_Tm_max,
            Tm_weight=target_probe_Tm_weight,
            Tm_parameters=self.target_probe_Tm_parameters,
            Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
            set_size_opt=set_size_opt,
            set_size_min=set_size_min,
            distance_between_oligos=distance_between_target_probes,
            n_sets=n_sets,
            max_graph_size=self.max_graph_size,
            n_attempts=self.n_attempts,
            heuristic=self.heuristic,
            heuristic_n_attempts=self.heuristic_n_attempts,
        )

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(name_database="4_db_target_probes_sets")
            dir_oligosets = oligo_database.write_oligosets_to_table()
            logging.info(
                f"Saved target probe database for step 4 (Specificity Filters) in directory {dir_database} and sets table in directory {dir_oligosets}"
            )

        return oligo_database

    def design_readout_probes(
        self,
        region_ids: list[str],
        file_readout_probe_table: str,
        file_codebook: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Design readout probes based on specified parameters.

        :param region_ids: List of region IDs for which readout probes are to be designed.
        :type region_ids: list[str]
        :param file_readout_probe_table: Path to the input readout probe table file.
        :type file_readout_probe_table: str
        :param file_codebook: Path to the input codebook file.
        :type file_codebook: str
        :return: A tuple containing the generated codebook and readout probe table.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        readout_probe_designer = ReadoutProbeDesigner(
            dir_output=self.dir_output,
            n_jobs=self.n_jobs,
        )
        if file_readout_probe_table:
            (
                readout_probe_table,
                n_channels,
                n_readout_probes_LR,
            ) = readout_probe_designer.load_readout_probe_table(
                file_readout_probe_table=file_readout_probe_table
            )
            logging.info(
                f"Loaded readout probes table from file and retrieved {n_channels} channels and {n_readout_probes_LR} L and R readout probes."
            )
        else:
            raise FeatureNotImplementedError(
                "Generation of readout probe table is not yet implemented. "
                "Please provide a file_readout_probe_table parameter."
            )

        if file_codebook:
            codebook = readout_probe_designer.load_codebook(file_codebook=file_codebook)
        else:
            codebook = readout_probe_designer.generate_codebook(
                n_regions=len(region_ids),
                n_channels=n_channels,
                n_readout_probes_LR=n_readout_probes_LR,
            )

        codebook.index = region_ids + [
            f"unassigned_barcode_{i+1}" for i in range(len(codebook.index) - len(region_ids))
        ]

        return codebook, readout_probe_table

    def assemble_hybridization_probes(
        self,
        target_probe_database: OligoDatabase,
        codebook: pd.DataFrame,
        readout_probe_table: pd.DataFrame,
        linker_sequence: str,
    ) -> OligoDatabase:
        """
        Assemble hybridization probes by combining target probes with readout probe sequences based on the codebook.

        :param target_probe_database: Database of target probes containing sequence and property information.
        :type target_probe_database: OligoDatabase
        :param codebook: A DataFrame containing barcodes for each region. Each row corresponds to a region,
            with columns representing bits in the barcode.
        :type codebook: pd.DataFrame
        :param readout_probe_table: A DataFrame containing readout probe sequences and their associated bit
            identifiers.
        :type readout_probe_table: pd.DataFrame
        :param linker_sequence: Sequence used to link target probes and readout probes in the encoding probe.
        :type linker_sequence: str
        :return: Database of assembled hybridization probes with properties and sequences.
        :rtype: OligoDatabase
        """
        region_ids = list(target_probe_database.database.keys())

        target_probe_database.set_database_sequence_types(
            [
                "sequence_target",
                "sequence_oligo_L",
                "sequence_oligo_R",
                "sequence_readout_probe_L",
                "sequence_readout_probe_R",
                "sequence_hybridization_probe_L",
                "sequence_hybridization_probe_R",
            ]
        )

        for region_id in region_ids:
            barcode = codebook.loc[region_id]
            bits = barcode[barcode == 1].index
            readout_probe_sequences = readout_probe_table.loc[bits, "readout_probe_sequence"]
            sequence_readout_probe_L = readout_probe_sequences.iloc[0]
            sequence_readout_probe_R = readout_probe_sequences.iloc[1]

            probe_ids = list(target_probe_database.database[region_id].keys())
            new_properties: dict[str, dict[str, str]] = {probe_id: {} for probe_id in probe_ids}

            for probe_id in probe_ids:
                new_properties[probe_id]["sequence_target"] = format_sequence(
                    database=target_probe_database,
                    property="target",
                    region_id=region_id,
                    oligo_id=probe_id,
                )
                new_properties[probe_id]["sequence_oligo_L"] = format_sequence(
                    database=target_probe_database,
                    property="oligo_L",
                    region_id=region_id,
                    oligo_id=probe_id,
                )
                new_properties[probe_id]["sequence_oligo_R"] = format_sequence(
                    database=target_probe_database,
                    property="oligo_R",
                    region_id=region_id,
                    oligo_id=probe_id,
                )
                new_properties[probe_id]["sequence_readout_probe_L"] = sequence_readout_probe_L
                new_properties[probe_id]["sequence_readout_probe_R"] = sequence_readout_probe_R

                new_properties[probe_id]["sequence_hybridization_probe_L"] = (
                    sequence_readout_probe_L
                    + str(Seq(linker_sequence).reverse_complement())
                    + format_sequence(
                        database=target_probe_database,
                        property="oligo_L",
                        region_id=region_id,
                        oligo_id=probe_id,
                    )
                )

                new_properties[probe_id]["sequence_hybridization_probe_R"] = (
                    format_sequence(
                        database=target_probe_database,
                        property="oligo_R",
                        region_id=region_id,
                        oligo_id=probe_id,
                    )
                    + str(Seq(linker_sequence).reverse_complement())
                    + sequence_readout_probe_R
                )

            target_probe_database.update_oligo_properties(new_properties)

        return target_probe_database

    def design_primers(
        self,
        forward_primer_sequence: str,
        reverse_primer_sequence: str,
    ) -> tuple[str, str]:
        """
        Design forward and reverse primers for the encoding probe database.

        :param forward_primer_sequence: Sequence of the forward primer.
        :type forward_primer_sequence: str
        :param reverse_primer_sequence: Sequence of the reverse primer.
        :type reverse_primer_sequence: str
        :type primer_secondary_structures_threshold_deltaG: float
        :return: A tuple containing the reverse primer sequence and the selected forward primer sequence.
        :rtype: tuple[str, str]
        """
        primer_designer = PrimerDesigner(
            dir_output=self.dir_output,
            n_jobs=self.n_jobs,
        )

        if forward_primer_sequence:
            forward_primer_sequence = forward_primer_sequence
        else:
            # generate forward primers
            raise FeatureNotImplementedError(
                "Forward primer generation is not yet implemented. "
                "Please provide a forward_primer_sequence parameter."
            )

        if reverse_primer_sequence:
            reverse_primer_sequence = reverse_primer_sequence
        else:
            # generate reverse primers
            raise FeatureNotImplementedError(
                "Reverse primer generation is not yet implemented. "
                "Please provide a reverse_primer_sequence parameter."
            )

        return reverse_primer_sequence, forward_primer_sequence

    def assemble_dna_template_probes(
        self,
        hybridization_probe_database: OligoDatabase,
        linker_sequence: str,
        forward_primer_sequence: str,
        reverse_primer_sequence: str,
    ) -> OligoDatabase:
        """
        Assemble DNA template probes by combining hybridization probes with forward and reverse primers.

        :param hybridization_probe_database: Database of hybridization probes containing sequence and property information.
        :type hybridization_probe_database: OligoDatabase
        :param linker_sequence: Sequence used to link hybridization probes and forward and reverse primers.
        :type linker_sequence: str
        :param forward_primer_sequence: Sequence of the forward primer.
        :type forward_primer_sequence: str
        :param reverse_primer_sequence: Sequence of the reverse primer.
        :type reverse_primer_sequence: str
        :return: Database of assembled DNA template probes with properties and sequences.
        :rtype: OligoDatabase
        """
        region_ids = list(hybridization_probe_database.database.keys())
        hybridization_probe_database.set_database_sequence_types(
            [
                "sequence_reverse_primer",
                "sequence_forward_primer",
                "sequence_dna_template_probe_L",
                "sequence_dna_template_probe_R",
            ]
        )

        for region_id in region_ids:
            probe_ids = list(hybridization_probe_database.database[region_id].keys())
            new_properties: dict[str, dict[str, str]] = {probe_id: {} for probe_id in probe_ids}

            for probe_id in probe_ids:
                new_properties[probe_id]["sequence_reverse_primer"] = reverse_primer_sequence
                new_properties[probe_id]["sequence_forward_primer"] = forward_primer_sequence

                new_properties[probe_id]["sequence_dna_template_probe_L"] = (
                    forward_primer_sequence
                    + str(
                        Seq(
                            format_sequence(
                                database=hybridization_probe_database,
                                property="sequence_oligo_L",
                                region_id=region_id,
                                oligo_id=probe_id,
                            )
                        ).reverse_complement()
                    )
                    + linker_sequence
                    + str(
                        Seq(
                            format_sequence(
                                database=hybridization_probe_database,
                                property="sequence_readout_probe_L",
                                region_id=region_id,
                                oligo_id=probe_id,
                            )
                        ).reverse_complement()
                    )
                    + reverse_primer_sequence
                )
                new_properties[probe_id]["sequence_dna_template_probe_R"] = (
                    forward_primer_sequence
                    + str(
                        Seq(
                            format_sequence(
                                database=hybridization_probe_database,
                                property="sequence_readout_probe_R",
                                region_id=region_id,
                                oligo_id=probe_id,
                            )
                        ).reverse_complement()
                    )
                    + linker_sequence
                    + str(
                        Seq(
                            format_sequence(
                                database=hybridization_probe_database,
                                property="sequence_oligo_R",
                                region_id=region_id,
                                oligo_id=probe_id,
                            )
                        ).reverse_complement()
                    )
                    + reverse_primer_sequence
                )

            hybridization_probe_database.update_oligo_properties(new_properties)

        return hybridization_probe_database

    def generate_output(
        self,
        probe_database: OligoDatabase,
        codebook: pd.DataFrame,
        readout_probe_table: pd.DataFrame,
        top_n_sets: int = 3,
    ) -> None:
        """
        Generate the final output files for the CycleHCR probe design pipeline.

        :param probe_database: Database of encoding probes with associated properties and sequences.
        :type probe_database: OligoDatabase
        :param codebook: Codebook used for the encoding probes.
        :type codebook: pd.DataFrame
        :param readout_probe_table: Table of readout probes used for the encoding probes.
        :type readout_probe_table: pd.DataFrame
        :param top_n_sets: Number of top probe sets to include in the output, defaults to 3.
        :type top_n_sets: int

        :return: None
        """
        # write codebook and readout probe table
        codebook.to_csv(os.path.join(self.dir_output, "codebook.tsv"), sep="\t", index_label="region_id")
        readout_probe_table.to_csv(os.path.join(self.dir_output, "readout_probes.tsv"), sep="\t")

        readout_probe_table_regions = []
        for region_id, barcode in codebook.iterrows():
            bits = barcode[barcode == 1].index
            readout_probe_info = readout_probe_table.loc[bits, :]
            readout_probe_info["region_id"] = region_id
            readout_probe_table_regions.append(readout_probe_info)
        readout_probe_table_regions_df = pd.concat(readout_probe_table_regions, axis=0)
        readout_probe_table_regions_df[
            ["region_id", "channel", "readout_probe_id", "L/R", "readout_probe_sequence"]
        ].to_csv(os.path.join(self.dir_output, "readout_probes_regions.tsv"), sep="\t", index=False)

        tm_nn_property: BaseProperty = TmNNProperty(
            Tm_parameters=self.target_probe_Tm_parameters,
            Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
        )

        calculator = PropertyCalculator(properties=[tm_nn_property])
        probe_database = calculator.apply(
            oligo_database=probe_database, sequence_type="sequence_oligo_L", n_jobs=self.n_jobs
        )
        probe_database = calculator.apply(
            oligo_database=probe_database, sequence_type="sequence_oligo_R", n_jobs=self.n_jobs
        )

        probe_database.write_oligosets_to_yaml(
            properties=self.output_properties,
            top_n_sets=top_n_sets,
            ascending=True,
            filename="cyclehcr_probes",
        )

        # write a second file that only contains order information
        yaml_dict_order: dict[str, dict] = {}

        for region_id in probe_database.database.keys():
            yaml_dict_order[region_id] = {}
            oligosets_region = probe_database.oligosets[region_id]
            oligosets_oligo_columns = [col for col in oligosets_region.columns if col.startswith("oligo_")]
            oligosets_score_columns = [col for col in oligosets_region.columns if col.startswith("score_")]

            oligosets_region.sort_values(by=oligosets_score_columns, ascending=True, inplace=True)
            oligosets_region = oligosets_region.head(top_n_sets)[oligosets_oligo_columns]
            oligosets_region.reset_index(inplace=True, drop=True)

            # iterate through all oligo sets
            for oligoset_idx, oligoset in oligosets_region.iterrows():
                oligoset_id = f"oligoset_{oligoset_idx + 1}"
                yaml_dict_order[region_id][oligoset_id] = {}
                for oligo_id in oligoset:
                    yaml_dict_order[region_id][oligoset_id][oligo_id] = {
                        "sequence_dna_template_probe_L": probe_database.get_oligo_property_value(
                            property="sequence_dna_template_probe_L",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                        "sequence_dna_template_probe_R": probe_database.get_oligo_property_value(
                            property="sequence_dna_template_probe_R",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                        "sequence_readout_probe_L": probe_database.get_oligo_property_value(
                            property="sequence_readout_probe_L",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                        "sequence_readout_probe_R": probe_database.get_oligo_property_value(
                            property="sequence_readout_probe_R",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                    }

        with open(os.path.join(self.dir_output, "cyclehcr_probes_order.yml"), "w") as outfile:
            yaml.dump(yaml_dict_order, outfile, default_flow_style=False, sort_keys=False)

        logging.info("--------------END PIPELINE--------------")


############################################
# CycleHCR Target Probe Designer
############################################


class TargetProbeDesigner:
    """
    A class for designing target probes for CycleHCR experiments.
    This class provides methods for creating, filtering, and scoring oligos based
    on specific properties and designing oligo sets for targeted probes.

    :param dir_output: Directory path where output files will be saved.
    :type dir_output: str
    :param n_jobs: Number of parallel jobs to use for processing.
    :type n_jobs: int
    """

    def __init__(self, dir_output: str, n_jobs: int) -> None:
        """Constructor for the TargetProbeDesigner class."""

        ##### create the output folder #####
        self.dir_output = os.path.abspath(dir_output)
        self.subdir_db_oligos = "db_target_probes"
        self.subdir_db_reference = "db_reference"

        self.n_jobs = n_jobs

    @pipeline_step_basic(step_name="Target Probe Generation - Create Database")
    def create_oligo_database(
        self,
        region_ids: list[str] | None,
        target_probe_L_probe_sequence_length: int,
        target_probe_gap_sequence_length: int,
        target_probe_R_probe_sequence_length: int,
        files_fasta_oligo_database: list[str],
        min_oligos_per_gene: int,
        isoform_consensus: float,
    ) -> OligoDatabase:
        """
        Creates an oligo database by generating sequences using a sliding window approach
        and filtering based on specified criteria.

        :param region_ids: List of region identifiers for which oligos should be generated.
                        If None, all regions in the input fasta file are used.
        :type region_ids: list[str] | None
        :param target_probe_L_probe_sequence_length: Length of the left probe sequence.
        :type target_probe_L_probe_sequence_length: int
        :param target_probe_gap_sequence_length: Length of the gap sequence.
        :type target_probe_gap_sequence_length: int
        :param target_probe_R_probe_sequence_length: Length of the right probe sequence.
        :type target_probe_R_probe_sequence_length: int
        :param files_fasta_oligo_database: List of FASTA files containing sequences for oligo generation.
        :type files_fasta_oligo_database: list[str]
        :param min_oligos_per_gene: Minimum number of oligos required per gene in the database.
        :type min_oligos_per_gene: int
        :param isoform_consensus: Threshold for isoform consensus filtering.
        :type isoform_consensus: float
        :return: The generated oligo database.
        :rtype: OligoDatabase
        """
        ##### creating the oligo sequences #####
        oligo_length = (
            target_probe_L_probe_sequence_length
            + target_probe_gap_sequence_length
            + target_probe_R_probe_sequence_length
        )
        oligo_sequences = OligoSequenceGenerator(dir_output=self.dir_output)
        oligo_fasta_file = oligo_sequences.create_sequences_sliding_window(
            files_fasta_in=files_fasta_oligo_database,
            length_interval_sequences=(oligo_length, oligo_length),
            region_ids=region_ids,
            n_jobs=self.n_jobs,
        )

        ##### creating the oligo database #####
        oligo_database = OligoDatabase(
            min_oligos_per_region=min_oligos_per_gene,
            write_regions_with_insufficient_oligos=True,
            max_entries_in_memory=self.n_jobs * 2 + 2,
            database_name=self.subdir_db_oligos,
            dir_output=self.dir_output,
            n_jobs=1,
        )
        oligo_database.load_database_from_fasta(
            files_fasta=oligo_fasta_file,
            database_overwrite=True,
            sequence_type="target",
            region_ids=region_ids,
        )
        # Set all sequence types that will be used in this pipeline
        oligo_database.set_database_sequence_types(["target", "oligo", "oligo_L", "oligo_R"])

        ##### pre-filter oligo database for certain properties #####
        isoform_consensus_property: BaseProperty = IsoformConsensusProperty()
        reverse_complement_sequence_property: BaseProperty = ReverseComplementSequenceProperty(
            sequence_type_reverse_complement="oligo"
        )

        calculator = PropertyCalculator(
            properties=[isoform_consensus_property, reverse_complement_sequence_property]
        )
        oligo_database = calculator.apply(
            oligo_database=oligo_database, sequence_type="target", n_jobs=self.n_jobs
        )
        oligo_database.filter_database_by_property_threshold(
            property_name="isoform_consensus",
            property_thr=isoform_consensus,
            remove_if_smaller_threshold=True,
        )

        ##### calculate probe pairs
        split_start_end = [
            (0, target_probe_L_probe_sequence_length),
            (
                target_probe_L_probe_sequence_length,
                target_probe_L_probe_sequence_length + target_probe_gap_sequence_length,
            ),
            (
                target_probe_L_probe_sequence_length + target_probe_gap_sequence_length,
                target_probe_L_probe_sequence_length
                + target_probe_gap_sequence_length
                + target_probe_R_probe_sequence_length,
            ),
        ]
        # Calculate split sequence using new PropertyCalculator pattern
        # first right then left sequence because we are splitting the oligo not the target sequence
        split_sequence_property: BaseProperty = SplitSequenceProperty(
            split_start_end=split_start_end,
            split_names=["oligo_R", "spacer", "oligo_L"],
        )

        calculator = PropertyCalculator(properties=[split_sequence_property])
        oligo_database = calculator.apply(
            oligo_database=oligo_database, sequence_type="oligo", n_jobs=self.n_jobs
        )

        dir = oligo_sequences.dir_output
        shutil.rmtree(dir) if os.path.exists(dir) else None

        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Pre-Filters")
        check_content_oligo_database(oligo_database)

        return oligo_database

    @pipeline_step_basic(step_name="Target Probe Generation - Property Filters")
    def filter_by_property(
        self,
        oligo_database: OligoDatabase,
        GC_content_min: float,
        GC_content_max: float,
        Tm_min: float,
        Tm_max: float,
        homopolymeric_base_n: dict,
        T_secondary_structure: float,
        secondary_structures_threshold_deltaG: float,
        Tm_parameters: dict,
        Tm_chem_correction_parameters: dict | None,
        Tm_salt_correction_parameters: dict | None,
    ) -> OligoDatabase:
        """
        Filter the oligo database based on various sequence properties.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param GC_content_min: Minimum acceptable GC content for oligos.
        :type GC_content_min: float
        :param GC_content_max: Maximum acceptable GC content for oligos.
        :type GC_content_max: float
        :param Tm_min: Minimum acceptable melting temperature (Tm) for oligos.
        :type Tm_min: float
        :param Tm_max: Maximum acceptable melting temperature (Tm) for oligos.
        :type Tm_max: float
        :param homopolymeric_base_n: Maximum allowable length of homopolymeric base runs.
        :type homopolymeric_base_n: dict
        :param T_secondary_structure: Temperature for secondary structure analysis.
        :type T_secondary_structure: float
        :param secondary_structures_threshold_deltaG: Threshold for secondary structure deltaG.
        :type secondary_structures_threshold_deltaG: float
        :param Tm_parameters: Parameters for melting temperature calculation.
        :type Tm_parameters: dict
        :param Tm_chem_correction_parameters: Parameters for chemical correction in Tm calculation.
        :type Tm_chem_correction_parameters: dict | None
        :param Tm_salt_correction_parameters: Parameters for salt correction in Tm calculation.
        :type Tm_salt_correction_parameters: dict | None
        :return: The filtered oligo database.
        :rtype: OligoDatabase
        """
        # define the filters
        # soft_masked_sequences = SoftMaskedSequenceFilter()
        hard_masked_sequences = HardMaskedSequenceFilter()
        gc_content = GCContentFilter(GC_content_min=GC_content_min, GC_content_max=GC_content_max)
        melting_temperature = MeltingTemperatureNNFilter(
            Tm_min=Tm_min,
            Tm_max=Tm_max,
            Tm_parameters=Tm_parameters,
            Tm_chem_correction_parameters=Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=Tm_salt_correction_parameters,
        )
        homopolymeric_runs = HomopolymericRunsFilter(
            base_n=homopolymeric_base_n,
        )
        secondary_sctructure = SecondaryStructureFilter(
            T=T_secondary_structure,
            thr_DG=secondary_structures_threshold_deltaG,
        )

        filters = [
            # soft_masked_sequences,
            hard_masked_sequences,
            homopolymeric_runs,
            gc_content,
            melting_temperature,
            secondary_sctructure,
        ]

        # initialize the preoperty filter class
        property_filter = PropertyFilter(filters=filters)

        # filter the database
        oligo_database = property_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo_L",
            n_jobs=self.n_jobs,
        )
        oligo_database = property_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo_R",
            n_jobs=self.n_jobs,
        )
        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Property Filters")
        check_content_oligo_database(oligo_database)

        return oligo_database

    @pipeline_step_basic(step_name="Target Probe Generation - Specificity Filters")
    def filter_by_specificity(
        self,
        oligo_database: OligoDatabase,
        files_fasta_reference_database: list[str],
        junction_region_size: int,
        junction_site: int,
        specificity_blastn_search_parameters: dict,
        specificity_blastn_hit_parameters: dict,
        cross_hybridization_blastn_search_parameters: dict,
        cross_hybridization_blastn_hit_parameters: dict,
    ) -> OligoDatabase:
        """
        Filter the oligo database based on sequence specificity to remove sequences that
        cross-hybridize to other oligos or hybridization to other genomic regions.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param files_fasta_reference_database: List of FASTA files containing reference sequences for specificity filtering.
        :type files_fasta_reference_database: list[str]
        :param junction_region_size: Size of the junction region for seed-based specificity filtering.
        :type junction_region_size: int
        :param junction_site: Position of the junction site within the oligo sequence.
        :type junction_site: int
        :param specificity_blastn_search_parameters: Parameters for BLASTN specificity search.
        :type specificity_blastn_search_parameters: dict
        :param specificity_blastn_hit_parameters: Parameters for filtering BLASTN specificity hits.
        :type specificity_blastn_hit_parameters: dict
        :param cross_hybridization_blastn_search_parameters: Parameters for BLASTN cross-hybridization search.
        :type cross_hybridization_blastn_search_parameters: dict
        :param cross_hybridization_blastn_hit_parameters: Parameters for filtering BLASTN cross-hybridization hits.
        :type cross_hybridization_blastn_hit_parameters: dict
        :return: The filtered oligo database.
        :rtype: OligoDatabase
        """
        ##### define reference database #####
        reference_database = ReferenceDatabase(
            database_name=self.subdir_db_reference, dir_output=self.dir_output
        )
        reference_database.load_database_from_file(
            files=files_fasta_reference_database, file_type="fasta", database_overwrite=False
        )

        ##### define specificity filters #####
        exact_matches = ExactMatchFilter(policy=RemoveAllFilterPolicy(), filter_name="oligo_exact_match")

        specificity: AlignmentSpecificityFilter
        if junction_region_size > 0:
            oligo_ids = oligo_database.get_oligoid_list()
            oligo_database.update_oligo_properties(
                new_oligo_property={oligo_id: {"junction_site": junction_site} for oligo_id in oligo_ids}
            )
            specificity = BlastNSeedregionSiteFilter(
                seedregion_size=junction_region_size,
                seedregion_site_name="junction_site",
                search_parameters=specificity_blastn_search_parameters,
                hit_parameters=specificity_blastn_hit_parameters,
                filter_name="oligo_blastn_specificity",
                dir_output=self.dir_output,
            )
        else:
            specificity = BlastNFilter(
                search_parameters=specificity_blastn_search_parameters,
                hit_parameters=specificity_blastn_hit_parameters,
                filter_name="oligo_blastn_specificity",
                dir_output=self.dir_output,
            )
        specificity.set_reference_database(reference_database=reference_database)

        ##### run specificity filters #####
        specificity_filter = SpecificityFilter(filters=[exact_matches, specificity])
        oligo_database = specificity_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo",
            n_jobs=self.n_jobs,
        )

        ##### define cross hybridization filter #####
        cross_hybridization_aligner_oligo_pair_L = BlastNFilter(
            remove_hits=True,
            search_parameters=cross_hybridization_blastn_search_parameters,
            hit_parameters=cross_hybridization_blastn_hit_parameters,
            filter_name="oligo_L_R_blastn_crosshybridization",
            dir_output=self.dir_output,
        )
        cross_hybridization_oligo_pair_L = CrossHybridizationFilter(
            policy=RemoveByLargerRegionFilterPolicy(),
            alignment_method=cross_hybridization_aligner_oligo_pair_L,
            sequence_type_reference="oligo_L",
            filter_name="oligo_L_R_blastn_crosshybridization",
            dir_output=self.dir_output,
        )
        cross_hybridization_aligner_oligo_pair_R = BlastNFilter(
            remove_hits=True,
            search_parameters=cross_hybridization_blastn_search_parameters,
            hit_parameters=cross_hybridization_blastn_hit_parameters,
            filter_name="oligo_L_R_blastn_crosshybridization",
            dir_output=self.dir_output,
        )
        cross_hybridization_oligo_pair_R = CrossHybridizationFilter(
            policy=RemoveByLargerRegionFilterPolicy(),
            alignment_method=cross_hybridization_aligner_oligo_pair_R,
            sequence_type_reference="oligo_R",
            filter_name="oligo_L_R_blastn_crosshybridization",
            dir_output=self.dir_output,
        )

        ##### run cross hybridization filter #####
        specificity_filter = SpecificityFilter(
            filters=[cross_hybridization_oligo_pair_L, cross_hybridization_oligo_pair_R]
        )
        oligo_database = specificity_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo_L",
            n_jobs=self.n_jobs,
        )
        oligo_database = specificity_filter.apply(
            oligo_database=oligo_database,
            sequence_type="oligo_R",
            n_jobs=self.n_jobs,
        )

        ##### remove all directories of intermediate steps #####
        for directory in [
            cross_hybridization_aligner_oligo_pair_L.dir_output,
            cross_hybridization_aligner_oligo_pair_R.dir_output,
            cross_hybridization_oligo_pair_L.dir_output,
            cross_hybridization_oligo_pair_R.dir_output,
            specificity.dir_output,
        ]:
            if os.path.exists(directory):
                shutil.rmtree(directory)

        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Specificity Filters")
        check_content_oligo_database(oligo_database)

        return oligo_database

    @pipeline_step_basic(step_name="Target Probe Generation - Set Selection")
    def create_oligo_sets(
        self,
        oligo_database: OligoDatabase,
        isoform_weight: float,
        Tm_max: float,
        Tm_weight: float,
        Tm_parameters: dict,
        Tm_chem_correction_parameters: dict | None,
        Tm_salt_correction_parameters: dict | None,
        set_size_opt: int,
        set_size_min: int,
        distance_between_oligos: int,
        n_sets: int,
        max_graph_size: int,
        n_attempts: int,
        heuristic: bool,
        heuristic_n_attempts: int,
    ) -> OligoDatabase:
        """
        Create optimal oligo sets based on weighted scoring criteria, distance constraints and selection policies.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param isoform_weight: Weight assigned to isoform specificity in scoring.
        :type isoform_weight: float
        :param Tm_max: Maximum acceptable melting temperature (Tm) for oligos.
        :type Tm_max: float
        :param Tm_weight: Weight assigned to melting temperature (Tm) in scoring.
        :type Tm_weight: float
        :param Tm_parameters: Parameters for Tm calculation.
        :type Tm_parameters: dict
        :param Tm_chem_correction_parameters: Parameters for chemical correction in Tm calculation.
        :type Tm_chem_correction_parameters: dict | None
        :param Tm_salt_correction_parameters: Parameters for salt correction in Tm calculation.
        :type Tm_salt_correction_parameters: dict | None
        :param set_size_opt: Optimal size for oligo sets.
        :type set_size_opt: int
        :param set_size_min: Minimum size for oligo sets.
        :type set_size_min: int
        :param distance_between_oligos: Minimum genomic distance between oligos in a set.
        :type distance_between_oligos: int
        :param n_sets: Number of oligo sets to generate.
        :type n_sets: int
        :param max_graph_size: Maximum size of the graph used in set selection.
        :type max_graph_size: int
        :param n_attempts: Maximum number of attempts for selecting oligo sets.
        :type n_attempts: int
        :param heuristic: Whether to apply heuristic methods in oligo set selection.
        :type heuristic: bool
        :param heuristic_n_attempts: Maximum number of attempts for heuristic selecting oligo sets.
        :type heuristic_n_attempts: int
        :return: The updated oligo database.
        :rtype: OligoDatabase
        """
        # Define all scorers
        isoform_consensus_scorer = IsoformConsensusScorer(normalize=True, score_weight=isoform_weight)
        Tm_scorer = DeviationFromOptimalTmScorer(
            Tm_opt=Tm_max,
            Tm_parameters=Tm_parameters,
            Tm_salt_correction_parameters=Tm_salt_correction_parameters,
            Tm_chem_correction_parameters=Tm_chem_correction_parameters,
            score_weight=Tm_weight,
        )

        oligos_scoring = OligoScoring(scorers=[isoform_consensus_scorer, Tm_scorer])

        # the higher the score the better, because we want to have on average oligos with high melting temperatures
        set_scoring = AverageSetScoring(ascending=False)

        # We change the processing dependent on the required number of probes in the probe sets
        # For small sets, we don't pre-filter and find the initial set by iterating
        # through all possible generated sets, which is faster than the max clique approximation.
        selection_policy: OligoSelectionPolicy
        if set_size_opt < 10:
            pre_filter = False
            clique_init_approximation = False
            selection_policy = GraphBasedSelectionPolicy(
                set_scoring=set_scoring,
                pre_filter=pre_filter,
                n_attempts=n_attempts,
                heuristic=heuristic,
                heuristic_n_attempts=heuristic_n_attempts,
                clique_init_approximation=clique_init_approximation,
            )
            base_log_parameters(
                {
                    "pre_filter": pre_filter,
                    "clique_init_approximation": clique_init_approximation,
                    "selection_policy": "Graph-Based",
                }
            )

        # For medium sized sets, we don't pre-filter but we apply the max clique approximation
        # to find an initial probe set faster.
        elif 10 < set_size_opt < 30:
            pre_filter = False
            clique_init_approximation = True
            selection_policy = GraphBasedSelectionPolicy(
                set_scoring=set_scoring,
                pre_filter=pre_filter,
                n_attempts=n_attempts,
                heuristic=heuristic,
                heuristic_n_attempts=heuristic_n_attempts,
                clique_init_approximation=clique_init_approximation,
            )
            base_log_parameters(
                {
                    "pre_filter": pre_filter,
                    "clique_init_approximation": clique_init_approximation,
                    "selection_policy": "Graph-Based",
                }
            )

        # For large sets, we apply the pre-filter which removes all probes from the
        # graph that are only part of cliques which are smaller than the minimum set size
        # and we apply the Greedy Selection Policy istead of the graph-based selection policy.
        else:
            pre_filter = True
            selection_policy = GreedySelectionPolicy(
                set_scoring=set_scoring,
                score_criteria=set_scoring.score_1,
                pre_filter=pre_filter,
                penalty=0.01,
                n_attempts=n_attempts,
            )
            base_log_parameters({"pre_filter": pre_filter, "selection_policy": "Greedy"})

        probeset_generator = OligosetGeneratorIndependentSet(
            selection_policy=selection_policy,
            oligos_scoring=oligos_scoring,
            set_scoring=set_scoring,
            max_oligos=max_graph_size,
            distance_between_oligos=distance_between_oligos,
        )
        oligo_database = probeset_generator.apply(
            oligo_database=oligo_database,
            sequence_type="oligo",
            set_size_opt=set_size_opt,
            set_size_min=set_size_min,
            n_sets=n_sets,
            n_jobs=self.n_jobs,
        )

        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Oligo Selection")
        check_content_oligo_database(oligo_database)

        return oligo_database


############################################
# CycleHCR Readout Probe Designer
############################################


class ReadoutProbeDesigner:
    """
    A class for designing CycleHCR readout probes.
    This class provides methods for creating, filtering, and scoring oligos based
    on specific properties and designing oligo sets for readout probes.

    :param dir_output: Directory path where output files will be saved.
    :type dir_output: str
    :param n_jobs: Number of parallel jobs to use for processing.
    :type n_jobs: int
    """

    def __init__(
        self,
        dir_output: str,
        n_jobs: int,
    ) -> None:
        """Constructor for the ReadoutProbeDesigner class."""

        ##### create the output folder #####
        self.dir_output = os.path.abspath(dir_output)

        self.n_jobs = n_jobs

    @pipeline_step_basic(step_name="Readout Probe Generation - Create Oligo Database")
    def create_oligo_database(
        self,
        oligo_length: int,
        oligo_base_probabilities: dict,
        initial_num_sequences: int,
    ) -> OligoDatabase:
        """
        Create an oligo database containing sequences of specific length and base probabilities.

        :param oligo_length: Length of the oligo sequences to generate.
        :type oligo_length: int
        :param oligo_base_probabilities: Dictionary specifying the base probabilities for each position in the oligo.
        :type oligo_base_probabilities: dict
        :param initial_num_sequences: Number of sequences to generate initially.
        :type initial_num_sequences: int
        :return: An OligoDatabase containing the generated oligo sequences.
        :rtype: OligoDatabase
        """
        raise FeatureNotImplementedError("Creation of oligo database is not yet implemented. ")

    @pipeline_step_basic(step_name="Readout Probe Generation - Set Selection")
    def create_oligo_sets(
        self,
        oligo_database: OligoDatabase,
    ) -> OligoDatabase:
        """
        Create oligo sets for the readout probes.

        :param oligo_database: The oligo database to create sets from.
        :type oligo_database: OligoDatabase
        :return: The oligo database with the sets.
        :rtype: OligoDatabase
        """
        raise FeatureNotImplementedError("Creation of oligo sets is not yet implemented. ")

    @pipeline_step_basic(step_name="Readout Probe Generation - Property Filters")
    def filter_by_property(
        self,
        oligo_database: OligoDatabase,
    ) -> OligoDatabase:
        """
        Filter the oligo database by property.

        :param oligo_database: The oligo database to filter.
        :type oligo_database: OligoDatabase
        :return: The filtered oligo database.
        :rtype: OligoDatabase
        """
        raise FeatureNotImplementedError("Filtering by property is not yet implemented. ")

    def filter_by_specificity(
        self,
        oligo_database: OligoDatabase,
    ) -> OligoDatabase:
        """
        Filter the oligo database by specificity.

        :param oligo_database: The oligo database to filter.
        :type oligo_database: OligoDatabase
        :return: The filtered oligo database.
        :rtype: OligoDatabase
        """
        raise FeatureNotImplementedError("Filtering by specificity is not yet implemented. ")

    def generate_codebook(self, n_regions: int, n_channels: int, n_readout_probes_LR: int) -> pd.DataFrame:
        """
        Generate a codebook (barcode matrix) for encoding multiple regions using CycleHCR readout probes.

        Each region is encoded using a unique barcode derived from valid probe/channel combinations.
        The function ensures that each barcode has two active bits, representing a left-right probe pairing.

        :param n_regions: Number of unique regions to encode in the codebook.
        :type n_regions: int
        :param n_channels: Number of fluorescence channels used in the experiment.
        :type n_channels: int
        :param n_readout_probes_LR: Total number of L/R readout probes per channel.
        :type n_readout_probes_LR: int
        :return: A pandas DataFrame containing the binary barcode matrix for all regions.
        :rtype: pd.DataFrame
        """

        def _generate_barcode(combination: tuple[int, int, int], codebook_size: int) -> list:
            index1 = ((n_channels * 2) * combination[0]) + (2 * combination[2])
            index2 = ((n_channels * 2) * combination[1]) + (2 * combination[2]) + 1
            barcode = np.zeros(codebook_size, dtype=np.int8)
            barcode[[index1, index2]] = 1
            return list(barcode)

        codebook_list = []
        codebook_size = n_channels * n_readout_probes_LR * 2

        combinations = list(
            itertools.product(
                list(range(n_readout_probes_LR)), list(range(n_readout_probes_LR)), list(range(n_channels))
            )
        )
        combinations = sorted(combinations, key=lambda t: (0 if t[0] == t[1] else 1, t[1]))
        codebook_size_max = len(combinations)

        if codebook_size_max < (2 * n_regions):
            raise ConfigurationError(
                f"The number of valid barcodes ({codebook_size_max}) is lower than the required number of readout probes ({2 * n_regions}) for {n_regions} regions. "
                f"Consider increasing the number of L/R readout probes or reducing the number of regions."
            )

        for combination in combinations[:n_regions]:
            barcode = _generate_barcode(
                combination=combination,
                codebook_size=codebook_size,
            )
            codebook_list.append(barcode)

        codebook: pd.DataFrame = pd.DataFrame(
            codebook_list, columns=[f"bit_{i+1}" for i in range(codebook_size)]
        )

        # Remove columns where all values are 0
        codebook = codebook.loc[:, (codebook != 0).any(axis=0)]

        return codebook

    def load_codebook(self, file_codebook: str) -> pd.DataFrame:
        """
        Load a codebook from a file.

        :param file_codebook: Path to the file containing the codebook.
        :type file_codebook: str
        :return: The codebook with region IDs as index.
        :rtype: pd.DataFrame
        :raises FileFormatError: If the codebook doesn't contain at least one bit column,
            at least one row, or if not all columns are named with "bit_*".
        """
        codebook = pd.read_csv(file_codebook, sep=None, engine="python", index_col="region_id")

        # Check for at least one column
        if len(codebook.columns) == 0:
            raise FileFormatError(f"Codebook file '{file_codebook}' must contain at least one column.")

        # Check that all columns start with "bit_"
        non_bit_columns = [col for col in codebook.columns if not str(col).startswith("bit_")]
        if len(non_bit_columns) > 0:
            raise FileFormatError(
                f"Codebook file '{file_codebook}' must have all columns named with 'bit_*'. "
                f"Found columns that don't match: {non_bit_columns}"
            )

        # Check for at least one data row (excluding empty rows)
        codebook_clean = codebook.dropna(how="all")
        if len(codebook_clean) == 0:
            raise FileFormatError(f"Codebook file '{file_codebook}' must contain at least one row with data.")

        return codebook

    def create_readout_probe_table(
        self,
        readout_probe_database: OligoDatabase,
        channels_ids: list,
    ) -> pd.DataFrame:
        """
        Create a readout probe table that maps bits to channels and readout probes.

        :param readout_probe_database: The database containing readout probes and their sequences.
        :type readout_probe_database: OligoDatabase
        :param channels_ids: List of channel identifiers to assign to the readout probes (e.g., fluorophore channels).
        :type channels_ids: list
        :param n_barcode_rounds: Number of barcode rounds.
        :type n_barcode_rounds: int
        :param n_pseudocolors: Number of pseudocolors.
        :type n_pseudocolors: int
        :return: DataFrame containing the readout probe table with columns for bit, channel, readout probe ID, and readout probe sequence.
        :rtype: pd.DataFrame
        """
        raise FeatureNotImplementedError("Generation of readout probe table is not yet implemented. ")

    def load_readout_probe_table(self, file_readout_probe_table: str) -> tuple[pd.DataFrame, int, int]:
        """
        Load and validate a table containing readout probe information.

        The input table must include the required columns: 'channel', 'readout_probe_id', 'L/R',
        and 'readout_probe_sequence'. The function also assigns bit labels if not present.

        :param file_readout_probe_table: Path to the CSV/TSV file containing the readout probe data.
        :type file_readout_probe_table: str
        :return: Tuple containing the formatted DataFrame, number of channels, and number of L/R probes per channel.
        :rtype: tuple[pd.DataFrame, int, int]
        """
        required_cols = ["channel", "readout_probe_id", "L/R", "readout_probe_sequence"]

        readout_probe_table = pd.read_csv(file_readout_probe_table, sep=None, engine="python")

        # Check if all required columns exist in readout_probe_table
        cols = set(readout_probe_table.columns)
        if not set(required_cols).issubset(cols):
            missing = set(required_cols) - cols
            raise FileFormatError(
                f"Readout probe table is missing required columns: {missing}. "
                f"Required columns are: {required_cols}."
            )

        if "bit" not in readout_probe_table.columns:
            readout_probe_table = readout_probe_table.sort_values(by=["readout_probe_id", "channel"])
            readout_probe_table.reset_index(inplace=True, drop=True)
            readout_probe_table["bit"] = "bit_" + (readout_probe_table.index + 1).astype(str)

        readout_probe_table.set_index("bit", inplace=True)
        readout_probe_table = readout_probe_table[required_cols]

        n_channels = len(readout_probe_table["channel"].unique())
        n_readout_probes_R = readout_probe_table["L/R"].value_counts()["R"]
        n_readout_probes_L = readout_probe_table["L/R"].value_counts()["L"]
        n_readout_probes_LR = int(min([n_readout_probes_R, n_readout_probes_L]) / n_channels)

        return readout_probe_table, n_channels, n_readout_probes_LR


############################################
# CycleHCR Primer Designer
############################################


class PrimerDesigner:
    """
    A class for designing CycleHCR primers.
    This class provides methods for creating, filtering, and scoring oligos based
    on specific properties and designing oligo sets for primers.

    Note: The implementation of primer design logic is not yet provided.

    :param dir_output: Directory path where output files will be saved.
    :type dir_output: str
    :param n_jobs: Number of parallel jobs to use for processing.
    :type n_jobs: int
    """

    def __init__(
        self,
        dir_output: str,
        n_jobs: int,
    ) -> None:
        """Constructor for the PrimerDesigner class."""

        ##### create the output folder #####
        self.dir_output = os.path.abspath(dir_output)
        self.n_jobs = n_jobs


############################################
# CycleHCR Probe Designer Pipeline
############################################


def main() -> None:
    """
    Main function for running the CycleHCRProbeDesigner pipeline. This function reads the configuration file,
    processes gene IDs, initializes the probe designer, sets developer parameters, and executes probe design
    and output generation steps.

    :param args: Command-line arguments parsed using the base parser. The arguments include:
        - config: Path to the configuration YAML file containing parameters for the pipeline.
    :type args: argparse.Namespace
    """
    logging.info("--------------START PIPELINE--------------")

    args = base_parser()

    ##### read the config file #####
    with open(args["config"], "r") as handle:
        config = yaml.safe_load(handle)

    ##### read the genes file #####
    if config["file_regions"] is None:
        warnings.warn(
            "No gene list file was provided! All genes from fasta file are used to generate the probes. This chioce can use a lot of resources."
        )
        gene_ids = None
    else:
        with open(config["file_regions"]) as handle:
            lines = handle.readlines()
            # ensure that the list contains unique gene ids
            gene_ids = list(set([line.rstrip() for line in lines]))

    ##### initialize probe designer pipeline #####
    pipeline = CycleHCRProbeDesigner(
        write_intermediate_steps=config["write_intermediate_steps"],
        dir_output=config["dir_output"],
        n_jobs=config["n_jobs"],
    )

    ##### set custom developer parameters #####
    pipeline.set_developer_parameters(
        target_probe_specificity_blastn_search_parameters=config[
            "target_probe_specificity_blastn_search_parameters"
        ],
        target_probe_specificity_blastn_hit_parameters=config[
            "target_probe_specificity_blastn_hit_parameters"
        ],
        target_probe_cross_hybridization_blastn_search_parameters=config[
            "target_probe_cross_hybridization_blastn_search_parameters"
        ],
        target_probe_cross_hybridization_blastn_hit_parameters=config[
            "target_probe_cross_hybridization_blastn_hit_parameters"
        ],
        target_probe_Tm_parameters=config["target_probe_Tm_parameters"],
        target_probe_Tm_chem_correction_parameters=config["target_probe_Tm_chem_correction_parameters"],
        target_probe_Tm_salt_correction_parameters=config["target_probe_Tm_salt_correction_parameters"],
        max_graph_size=config["max_graph_size"],
        n_attempts=config["n_attempts"],
        heuristic=config["heuristic"],
        heuristic_n_attempts=config["heuristic_n_attempts"],
    )

    ##### design probes #####
    target_probe_database = pipeline.design_target_probes(
        region_ids=gene_ids,
        files_fasta_target_probe_database=config["files_fasta_target_probe_database"],
        files_fasta_reference_database_target_probe=config["files_fasta_reference_database_target_probe"],
        target_probe_isoform_consensus=config["target_probe_isoform_consensus"],
        target_probe_L_probe_sequence_length=config["target_probe_L_probe_sequence_length"],
        target_probe_gap_sequence_length=config["target_probe_gap_sequence_length"],
        target_probe_R_probe_sequence_length=config["target_probe_R_probe_sequence_length"],
        target_probe_GC_content_min=config["target_probe_GC_content_min"],
        target_probe_GC_content_max=config["target_probe_GC_content_max"],
        target_probe_Tm_min=config["target_probe_Tm_min"],
        target_probe_Tm_max=config["target_probe_Tm_max"],
        target_probe_homopolymeric_base_n=config["target_probe_homopolymeric_base_n"],
        target_probe_T_secondary_structure=config["target_probe_T_secondary_structure"],
        target_probe_secondary_structures_threshold_deltaG=config[
            "target_probe_secondary_structures_threshold_deltaG"
        ],
        target_probe_junction_region_size=config["target_probe_junction_region_size"],
        target_probe_Tm_weight=config["target_probe_Tm_weight"],
        target_probe_isoform_weight=config["target_probe_isoform_weight"],
        set_size_opt=config["set_size_opt"],
        set_size_min=config["set_size_min"],
        distance_between_target_probes=config["distance_between_target_probes"],
        n_sets=config["n_sets"],
    )

    codebook, readout_probe_table = pipeline.design_readout_probes(
        region_ids=list(target_probe_database.database.keys()),
        file_readout_probe_table=config["file_readout_probe_table"],
        file_codebook=config["file_codebook"],
    )

    hybridization_probe_database = pipeline.assemble_hybridization_probes(
        target_probe_database=target_probe_database,
        codebook=codebook,
        readout_probe_table=readout_probe_table,
        linker_sequence=config["linker_sequence"],
    )

    reverse_primer_sequence, forward_primer_sequence = pipeline.design_primers(
        forward_primer_sequence=config["forward_primer_sequence"],
        reverse_primer_sequence=config["reverse_primer_sequence"],
    )

    final_probe_database = pipeline.assemble_dna_template_probes(
        hybridization_probe_database=hybridization_probe_database,
        linker_sequence=config["linker_sequence"],
        forward_primer_sequence=forward_primer_sequence,
        reverse_primer_sequence=reverse_primer_sequence,
    )

    pipeline.generate_output(
        probe_database=final_probe_database,
        codebook=codebook,
        readout_probe_table=readout_probe_table,
        top_n_sets=config["top_n_sets"],
    )

    logging.info("--------------END PIPELINE--------------")


if __name__ == "__main__":
    main()
