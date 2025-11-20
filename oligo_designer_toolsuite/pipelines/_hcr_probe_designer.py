############################################
# imports
############################################

import logging
import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
import yaml
from Bio.SeqUtils import MeltingTemp as mt

from oligo_designer_toolsuite._exceptions import (
    FeatureNotImplementedError,
    FileFormatError,
)
from oligo_designer_toolsuite.database import OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_efficiency_filter import (
    AverageSetScoring,
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
# HCR Probe Designer
############################################


class HcrProbeDesigner:
    """
    A class for designing encoding probes for the HCR experiments.

    A HCR encoding probe is a fluorescent probe that contains a 92-nt targeting sequence (divided into
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
        """Constructor for the HcrProbeDesigner class."""

        # create the output folder
        self.dir_output = os.path.abspath(dir_output)
        Path(self.dir_output).mkdir(parents=True, exist_ok=True)

        # setup logger
        setup_logging(
            dir_output=self.dir_output,
            pipeline_name="hcr_probe_designer",
            log_start_message=True,
        )

        ##### set class parameters #####
        self.write_intermediate_steps = write_intermediate_steps
        self.n_jobs = n_jobs

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
                "sequence_linker",
                "sequence_initiator_probe_L",
                "sequence_initiator_probe_R",
                "sequence_hybridization_probe_L",
                "sequence_hybridization_probe_R",
                "TmNN_sequence_target_L",
                "TmNN_sequence_target_R",
                "isoform_consensus",
            ]
        else:
            self.output_properties = output_properties

    def set_developer_parameters(
        self,
        target_probe_specificity_blastn_search_parameters: dict,
        target_probe_specificity_blastn_hit_parameters: dict,
        target_probe_cross_hybridization_blastn_search_parameters: dict,
        target_probe_cross_hybridization_blastn_hit_parameters: dict,
        target_probe_Tm_parameters: dict,
        target_probe_Tm_chem_correction_parameters: dict | None,
        target_probe_Tm_salt_correction_parameters: dict | None,
        max_graph_size: int,
        n_attempts: int,
        heuristic: bool,
        heuristic_n_attempts: int,
    ) -> None:
        """
        Set developer-specific parameters for HCR probe designer pipeline.
        These parameters can be used to customize and fine-tune the pipeline.

        :param target_probe_specificity_blastn_search_parameters: Parameters for the BlastN specificity
            search for target probes.
        :type target_probe_specificity_blastn_search_parameters: dict
        :param target_probe_specificity_blastn_hit_parameters: Parameters for filtering BlastN hits
            for target probe specificity.
        :type target_probe_specificity_blastn_hit_parameters: dict
        :param target_probe_cross_hybridization_blastn_search_parameters: Parameters for the BlastN
            cross-hybridization search for target probes.
        :type target_probe_cross_hybridization_blastn_search_parameters: dict
        :param target_probe_cross_hybridization_blastn_hit_parameters: Parameters for filtering
            BlastN hits for target probe cross-hybridization.
        :type target_probe_cross_hybridization_blastn_hit_parameters: dict
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
        :param max_graph_size: Maximum size of the graph used in set selection.
        :type max_graph_size: int
        :param n_attempts: Maximum number of attempts for selecting oligo sets.
        :type n_attempts: int
        :param heuristic: Whether to apply heuristic methods in oligo set selection.
        :type heuristic: bool
        :param heuristic_n_attempts: Maximum number of attempts for heuristic selecting oligo sets.
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
        region_ids: list[str] | None,
        target_probe_isoform_consensus: float,
        target_probe_L_probe_sequence_length: int,
        target_probe_gap_sequence_length: int,
        target_probe_R_probe_sequence_length: int,
        target_probe_GC_content_min: float,
        target_probe_GC_content_max: float,
        target_probe_Tm_min: float,
        target_probe_Tm_max: float,
        target_probe_homopolymeric_base_n: dict,
        target_probe_T_secondary_structure: float,
        target_probe_secondary_structures_threshold_deltaG: float,
        target_probe_junction_region_size: int,
        target_probe_isoform_weight: float,
        set_size_opt: int,
        set_size_min: int,
        distance_between_target_probes: int,
        n_sets: int,
    ) -> OligoDatabase:
        """
        Design target probes based on specified parameters, including property and specificity filters.
        The designed probes are organized into sets based on customizable constraints.

        :param files_fasta_target_probe_database: List of input FASTA files for the target probe database.
        :type files_fasta_target_probe_database: list[str]
        :param files_fasta_reference_database_target_probe: List of input FASTA files for the reference database.
        :type files_fasta_reference_database_target_probe: list[str]
        :param region_ids: List of region IDs to target, or None to target all regions.
        :type region_ids: list[str]
        :param target_probe_isoform_consensus: Isoform consensus threshold for filtering.
        :type target_probe_isoform_consensus: float
        :param target_probe_L_probe_sequence_length: Length of the left probe sequence.
        :type target_probe_L_probe_sequence_length: int
        :param target_probe_gap_sequence_length: Length of the gap sequence between left and right probes.
        :type target_probe_gap_sequence_length: int
        :param target_probe_R_probe_sequence_length: Length of the right probe sequence.
        :type target_probe_R_probe_sequence_length: int
        :param target_probe_GC_content_min: Minimum GC content for target probes.
        :type target_probe_GC_content_min: float
        :param target_probe_GC_content_max: Maximum GC content for target probes.
        :type target_probe_GC_content_max: float
        :param target_probe_homopolymeric_base_n: Maximum allowed homopolymeric runs for each nucleotide.
        :type target_probe_homopolymeric_base_n: dict[str, int]
        :param target_probe_T_secondary_structure: Threshold temperature for secondary structure evaluation.
        :type target_probe_T_secondary_structure: float
        :param target_probe_secondary_structures_threshold_deltaG: DeltaG threshold for secondary structure stability.
        :type target_probe_secondary_structures_threshold_deltaG: float
        :param target_probe_junction_region_size: Size of the junction region for specificity filtering.
        :type target_probe_junction_region_size: int
        :param target_probe_isoform_weight: Weight for isoform consensus in probe scoring.
        :type target_probe_isoform_weight: float
        :param set_size_opt: Optimal size of oligo sets.
        :type set_size_opt: int
        :param set_size_min: Minimum size of oligo sets.
        :type set_size_min: int
        :param distance_between_target_probes: Minimum genomic distance between probes in a set.
        :type distance_between_target_probes: int
        :param n_sets: Number of oligo sets to generate.
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

    def design_initiators(
        self,
        region_ids: list[str],
        file_initiator_table: str,
        file_codebook: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Design initiators based on specified parameters.

        :param region_ids: List of region IDs for which initiators are to be designed.
        :type region_ids: list[str]
        :param file_initiator_table: Path to the input initiator table file.
        :type file_initiator_table: str
        :param file_codebook: Path to the input codebook file.
        :type file_codebook: str
        :return: A tuple containing the generated codebook and initiator table.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        initiator_designer = InitiatorDesigner(
            dir_output=self.dir_output,
            n_jobs=self.n_jobs,
        )
        if file_initiator_table:
            initiator_table = initiator_designer.load_initiator_table(
                file_initiator_table=file_initiator_table
            )
            logging.info(f"Loaded initiator table from file and retrieved {len(initiator_table)} initiators.")
        else:
            raise FeatureNotImplementedError(
                "Generation of initiator table is not yet implemented. "
                "Please provide a file_initiator_table parameter."
            )

        if file_codebook:
            codebook = initiator_designer.load_codebook(file_codebook=file_codebook)
        else:
            raise FeatureNotImplementedError(
                "Generation of codebook is not yet implemented. " "Please provide a file_codebook parameter."
            )

        # Check if all region_ids are in the codebook
        missing_region_ids = set(region_ids) - set(codebook.index)
        if len(missing_region_ids) > 0:
            raise FileFormatError(
                f"Codebook is missing the following region IDs: {sorted(missing_region_ids)}. "
                f"Codebook contains {len(codebook)} regions: {sorted(codebook.index.tolist())}"
            )

        return codebook, initiator_table

    def assemble_hybridization_probes(
        self,
        target_probe_database: OligoDatabase,
        codebook: pd.DataFrame,
        initiator_table: pd.DataFrame,
        linker_sequence: str,
    ) -> OligoDatabase:
        """
        Assemble hybridization probes by combining target probes with initiator sequences based on the codebook.

        :param target_probe_database: Database of target probes containing sequence and property information.
        :type target_probe_database: OligoDatabase
        :param codebook: A DataFrame containing barcodes for each region. Each row corresponds to a region,
            with columns representing bits in the barcode.
        :type codebook: pd.DataFrame
        :param initiator_table: A DataFrame containing initiator sequences and their associated bit
            identifiers.
        :type initiator_table: pd.DataFrame
        :param linker_sequence: Sequence used to link target probes and initiators in the encoding probe.
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
                "sequence_linker",
                "sequence_initiator_L",
                "sequence_initiator_R",
                "sequence_hybridization_probe_L",
                "sequence_hybridization_probe_R",
            ]
        )

        for region_id in region_ids:
            barcode = codebook.loc[region_id]
            bits = barcode[barcode == 1].index
            sequence_initiator_L = initiator_table.loc[bits, "initiator_L_sequence"].iloc[0]
            sequence_initiator_R = initiator_table.loc[bits, "initiator_R_sequence"].iloc[0]

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
                new_properties[probe_id]["sequence_linker"] = linker_sequence
                new_properties[probe_id]["sequence_initiator_L"] = sequence_initiator_L
                new_properties[probe_id]["sequence_initiator_R"] = sequence_initiator_R

                new_properties[probe_id]["sequence_hybridization_probe_L"] = (
                    sequence_initiator_L
                    + linker_sequence
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
                    + linker_sequence
                    + sequence_initiator_R
                )

            target_probe_database.update_oligo_properties(new_properties)

        return target_probe_database

    def generate_output(
        self,
        probe_database: OligoDatabase,
        codebook: pd.DataFrame,
        initiator_table: pd.DataFrame,
        top_n_sets: int = 3,
    ) -> None:
        """
        Generate the final output files for the HCR probe design pipeline.

        :param probe_database: Database of encoding probes with associated properties and sequences.
        :type probe_database: OligoDatabase
        :param codebook: Codebook used for the encoding probes.
        :type codebook: pd.DataFrame
        :param initiator_table: Table of initiators used for the encoding probes.
        :type initiator_table: pd.DataFrame
        :param top_n_sets: Number of top probe sets to include in the output, defaults to 3.
        :type top_n_sets: int

        :return: None
        """
        # write codebook and readout probe table
        codebook.to_csv(os.path.join(self.dir_output, "codebook.tsv"), sep="\t", index_label="region_id")
        initiator_table.to_csv(os.path.join(self.dir_output, "initiators.tsv"), sep="\t")

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
            filename="hcr_probes",
        )

        # write a second file that only contains order information
        yaml_dict_order: dict[str, dict] = {}
        csv_table_order = list()

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
                        "sequence_hybridization_probe_L": probe_database.get_oligo_property_value(
                            property="sequence_hybridization_probe_L",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                        "sequence_hybridization_probe_R": probe_database.get_oligo_property_value(
                            property="sequence_hybridization_probe_R",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                        "sequence_initiator_L": probe_database.get_oligo_property_value(
                            property="sequence_initiator_L",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                        "sequence_initiator_R": probe_database.get_oligo_property_value(
                            property="sequence_initiator_R",
                            region_id=region_id,
                            oligo_id=oligo_id,
                            flatten=True,
                        ),
                    }
                    csv_table_order.append(
                        {
                            "region_id": region_id,
                            "oligoset_id": oligoset_id,
                            "oligo_id": oligo_id,
                            "initiator_L": probe_database.get_oligo_property_value(
                                property="sequence_initiator_L",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "linker_L": probe_database.get_oligo_property_value(
                                property="sequence_linker",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "sequence_oligo_L": probe_database.get_oligo_property_value(
                                property="sequence_oligo_L",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "sequence_oligo_R": probe_database.get_oligo_property_value(
                                property="sequence_oligo_R",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "linker_R": probe_database.get_oligo_property_value(
                                property="sequence_linker",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "initiator_R": probe_database.get_oligo_property_value(
                                property="sequence_initiator_R",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "sequence_hybridization_probe_L": probe_database.get_oligo_property_value(
                                property="sequence_hybridization_probe_L",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                            "sequence_hybridization_probe_R": probe_database.get_oligo_property_value(
                                property="sequence_hybridization_probe_R",
                                region_id=region_id,
                                oligo_id=oligo_id,
                                flatten=True,
                            ),
                        }
                    )

        with open(os.path.join(self.dir_output, "hcr_probes_order.yml"), "w") as outfile:
            yaml.dump(yaml_dict_order, outfile, default_flow_style=False, sort_keys=False)

        csv_table_order_df = pd.DataFrame(csv_table_order)
        csv_table_order_df.to_csv(
            os.path.join(self.dir_output, "hcr_probes_order.tsv"), sep="\t", index=False
        )

        logging.info("--------------END PIPELINE--------------")


############################################
# HCR Target Probe Designer
############################################


class TargetProbeDesigner:
    """
    A class for designing target probes for HCR experiments.
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
            split_names=["oligo_L", "spacer", "oligo_R"],
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
        oligos_scoring = OligoScoring(scorers=[isoform_consensus_scorer])
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
# HCR Initiator Designer
############################################


class InitiatorDesigner:
    """
    A class for designing HCR initiators.
    This class provides methods for creating, filtering, and scoring oligos based
    on specific properties and designing oligo sets for initiators.

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
        """Constructor for the InitiatorDesigner class."""

        ##### create the output folder #####
        self.dir_output = os.path.abspath(dir_output)
        self.n_jobs = n_jobs

    @pipeline_step_basic(step_name="Initiator Generation - Create Oligo Database")
    def create_oligo_database(self) -> None:
        """
        Create an oligo database containing sequences of specific length and base probabilities.
        """
        raise FeatureNotImplementedError("Creation of oligo database is not yet implemented. ")

    @pipeline_step_basic(step_name="Initiator Generation - Set Selection")
    def create_oligo_sets(self) -> None:
        """
        Create oligo sets for the initiators.
        """
        raise FeatureNotImplementedError("Creation of oligo sets is not yet implemented. ")

    @pipeline_step_basic(step_name="Initiator Generation - Property Filters")
    def filter_by_property(self) -> None:
        """
        Filter the oligo database by property.
        """
        raise FeatureNotImplementedError("Filtering by property is not yet implemented. ")

    @pipeline_step_basic(step_name="Initiator Generation - Specificity Filters")
    def filter_by_specificity(self) -> None:
        """
        Filter the oligo database by specificity.
        """
        raise FeatureNotImplementedError("Filtering by specificity is not yet implemented. ")

    def generate_codebook(
        self,
    ) -> None:
        """
        Generate a codebook for the initiators.
        """

        raise FeatureNotImplementedError("Generation of codebook is not yet implemented. ")

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

    def create_initiator_table(self) -> None:
        """
        Create a initiator table that maps bits to initiators.
        """
        raise FeatureNotImplementedError("Generation of initiator table is not yet implemented. ")

    def load_initiator_table(self, file_initiator_table: str) -> pd.DataFrame:
        """
        Load and validate a table containing initiator information.

        The input table must include the required columns: 'bit', 'initiator_L_sequence',
        and 'initiator_R_sequence'.

        :param file_initiator_table: Path to the CSV/TSV file containing the initiator data.
        :type file_initiator_table: str
        :return: The formatted DataFrame.
        :rtype: pd.DataFrame
        """
        required_cols = ["bit", "initiator_L_sequence", "initiator_R_sequence"]

        initiator_table = pd.read_csv(file_initiator_table, sep=None, engine="python")

        # Check if all required columns exist in readout_probe_table
        cols = set(initiator_table.columns)
        if not set(required_cols).issubset(cols):
            missing = set(required_cols) - cols
            raise FileFormatError(
                f"Initiator table is missing required columns: {missing}. "
                f"Required columns are: {required_cols}."
            )

        initiator_table = initiator_table[required_cols]
        initiator_table = initiator_table.set_index("bit")

        return initiator_table


############################################
# HCR Probe Designer Pipeline
############################################


def main() -> None:
    """
    Main function for running the HCRProbeDesigner pipeline. This function reads the configuration file,
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
    pipeline = HcrProbeDesigner(
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
        target_probe_isoform_weight=config["target_probe_isoform_weight"],
        set_size_opt=config["set_size_opt"],
        set_size_min=config["set_size_min"],
        distance_between_target_probes=config["distance_between_target_probes"],
        n_sets=config["n_sets"],
    )

    codebook, initiator_table = pipeline.design_initiators(
        region_ids=list(target_probe_database.database.keys()),
        file_initiator_table=config["file_initiator_table"],
        file_codebook=config["file_codebook"],
    )

    hybridization_probe_database = pipeline.assemble_hybridization_probes(
        target_probe_database=target_probe_database,
        codebook=codebook,
        initiator_table=initiator_table,
        linker_sequence=config["linker_sequence"],
    )

    pipeline.generate_output(
        probe_database=hybridization_probe_database,
        codebook=codebook,
        initiator_table=initiator_table,
        top_n_sets=config["top_n_sets"],
    )

    logging.info("--------------END PIPELINE--------------")


if __name__ == "__main__":
    main()
