############################################
# imports
############################################

import os
import shutil
import warnings
from typing import Optional

import yaml
from Bio import SeqIO
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from oligo_designer_toolsuite.database import OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_specificity_filter import (
    BlastNFilter,
    BlastNSeedregionLigationsiteFilter,
    SpecificityFilter,
)
from oligo_designer_toolsuite.pipelines._scrinshot_iss_probe_designer import (
    ScrinshotISSProbeDesigner,
)
from oligo_designer_toolsuite.pipelines._scrinshot_probe_designer import TargetProbeDesigner
from oligo_designer_toolsuite.pipelines._utils import (
    base_parser,
    check_content_oligo_database,
    pipeline_step_basic,
)
from oligo_designer_toolsuite.sequence_generator import OligoSequenceGenerator


class BlastNZeroHitMixin:
    """Alignment filter mixin that removes candidates with any configured BLAST hit."""

    def apply(
        self,
        sequence_type: str,
        oligo_database: OligoDatabase,
        reference_database: ReferenceDatabase,
        n_jobs: int = 1,
    ) -> OligoDatabase:
        consider_hits_from_input_region = True

        file_reference = reference_database.write_database_to_fasta(
            filename=f"db_reference_{self.filter_name}"
        )
        file_index = self._create_index(file_reference=file_reference, n_jobs=n_jobs)

        region_ids = list(oligo_database.database.keys())
        name = " ".join(string.capitalize() for string in self.filter_name.split("_"))
        with joblib_progress(description=f"Specificity Filter: {name}", total=len(region_ids)):
            Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
                delayed(self._apply_region)(
                    sequence_type=sequence_type,
                    oligo_database=oligo_database,
                    file_index=file_index,
                    region_id=region_id,
                    consider_hits_from_input_region=consider_hits_from_input_region,
                )
                for region_id in region_ids
            )

        os.remove(file_reference)
        self._remove_index(file_index)

        return oligo_database


class BlastNZeroHitFilter(BlastNZeroHitMixin, BlastNFilter):
    """BLASTN filter where every accepted hit removes the query candidate."""


class BlastNZeroHitSeedregionLigationsiteFilter(
    BlastNZeroHitMixin, BlastNSeedregionLigationsiteFilter
):
    """BLASTN ligation-seed filter where every accepted hit removes the query candidate."""


class CustomSequenceTargetProbeDesigner(TargetProbeDesigner):
    """Target probe designer for coordinate-free single-sequence FASTA inputs."""

    @pipeline_step_basic(step_name="Create Custom Sequence Database")
    def create_oligo_database_custom_sequence(
        self,
        gene_ids: list,
        oligo_length_min: int,
        oligo_length_max: int,
        files_fasta_oligo_database: list[str],
        min_oligos_per_gene: int,
    ) -> OligoDatabase:
        oligo_sequences = OligoSequenceGenerator(dir_output=self.dir_output)
        oligo_fasta_file = oligo_sequences.create_sequences_sliding_window(
            files_fasta_in=files_fasta_oligo_database,
            length_interval_sequences=(oligo_length_min, oligo_length_max),
            region_ids=gene_ids,
            n_jobs=self.n_jobs,
        )

        oligo_database = OligoDatabase(
            min_oligos_per_region=min_oligos_per_gene,
            write_regions_with_insufficient_oligos=True,
            lru_db_max_in_memory=self.n_jobs * 2 + 2,
            database_name=self.subdir_db_probes,
            dir_output=self.dir_output,
            n_jobs=1,
        )
        oligo_database.load_database_from_fasta(
            files_fasta=oligo_fasta_file,
            database_overwrite=True,
            sequence_type="target",
            region_ids=gene_ids,
        )
        oligo_database.remove_regions_with_insufficient_oligos(pipeline_step="Pre-Filters")

        dir_annotation = oligo_sequences.dir_output
        shutil.rmtree(dir_annotation) if os.path.exists(dir_annotation) else None

        return oligo_database

    @pipeline_step_basic(step_name="Transcriptome Zero-Hit Specificity Filters")
    def filter_by_transcriptome_zero_hits(
        self,
        oligo_database: OligoDatabase,
        files_fasta_reference_database: list[str],
        specificity_blastn_search_parameters: dict,
        specificity_blastn_hit_parameters: dict,
        ligation_region_size: int,
        arm_Tm_dif_max: int,
        arm_length_min: int,
        arm_Tm_min: float,
        arm_Tm_max: float,
        Tm_parameters: dict,
        Tm_chem_correction_parameters: dict,
        Tm_salt_correction_parameters: dict,
    ) -> OligoDatabase:
        reference_database = ReferenceDatabase(
            database_name=self.subdir_db_reference, dir_output=self.dir_output
        )
        reference_database.load_database_from_fasta(
            files_fasta=files_fasta_reference_database, database_overwrite=False
        )

        oligo_database = self.oligo_attributes_calculator.calculate_padlock_arms(
            oligo_database=oligo_database,
            arm_length_min=arm_length_min,
            arm_Tm_dif_max=arm_Tm_dif_max,
            arm_Tm_min=arm_Tm_min,
            arm_Tm_max=arm_Tm_max,
            Tm_parameters=Tm_parameters,
            Tm_chem_correction_parameters=Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=Tm_salt_correction_parameters,
        )

        if ligation_region_size > 0:
            specificity = BlastNZeroHitSeedregionLigationsiteFilter(
                seedregion_size=ligation_region_size,
                search_parameters=specificity_blastn_search_parameters,
                hit_parameters=specificity_blastn_hit_parameters,
                filter_name="blastn_transcriptome_zero_hit",
                dir_output=self.dir_output,
            )
        else:
            specificity = BlastNZeroHitFilter(
                search_parameters=specificity_blastn_search_parameters,
                hit_parameters=specificity_blastn_hit_parameters,
                filter_name="blastn_transcriptome_zero_hit",
                dir_output=self.dir_output,
            )

        specificity_filter = SpecificityFilter(filters=[specificity])
        oligo_database = specificity_filter.apply(
            sequence_type="oligo",
            oligo_database=oligo_database,
            reference_database=reference_database,
            n_jobs=self.n_jobs,
        )

        for directory in [reference_database.dir_output, specificity.dir_output]:
            if os.path.exists(directory):
                shutil.rmtree(directory)

        return oligo_database


class CustomSequenceScrinshotISSProbeDesigner(ScrinshotISSProbeDesigner):
    """
    SCRINSHOT ISS workflow for non-transcriptome targets such as transgenes or reporters.

    Candidate target sites are generated from a custom FASTA sequence, filtered by
    standard SCRINSHOT sequence-property and padlock-arm rules, then removed if they
    have any configured BLAST hit in the reference transcriptome.
    """

    def design_target_probes(
        self,
        files_fasta_target_probe_database: list,
        files_fasta_reference_database_target_probe: list,
        gene_ids: list = None,
        target_probe_length_min: int = 40,
        target_probe_length_max: int = 45,
        target_probe_isoform_weight: float = 0,
        target_probe_GC_content_min: float = 40,
        target_probe_GC_content_opt: float = 50,
        target_probe_GC_content_max: float = 60,
        target_probe_GC_weight: float = 1,
        target_probe_Tm_min: float = 65,
        target_probe_Tm_opt: float = 70,
        target_probe_Tm_max: float = 75,
        target_probe_Tm_weight: float = 1,
        target_probe_homopolymeric_base_n: dict = {"A": 5, "T": 5, "C": 5, "G": 5},
        detection_oligo_min_thymines: int = 2,
        detection_oligo_length_min: int = 15,
        detection_oligo_length_max: int = 40,
        target_probe_padlock_arm_length_min: int = 10,
        target_probe_padlock_arm_Tm_dif_max: float = 2,
        target_probe_padlock_arm_Tm_min: float = 50,
        target_probe_padlock_arm_Tm_max: float = 60,
        target_probe_ligation_region_size: int = 5,
        set_size_min: int = 3,
        set_size_opt: int = 5,
        distance_between_target_probes: int = 0,
        n_sets: int = 100,
    ) -> OligoDatabase:
        target_probe_designer = CustomSequenceTargetProbeDesigner(self.dir_output, self.n_jobs)

        oligo_database = target_probe_designer.create_oligo_database_custom_sequence(
            gene_ids=gene_ids,
            oligo_length_min=target_probe_length_min,
            oligo_length_max=target_probe_length_max,
            files_fasta_oligo_database=files_fasta_target_probe_database,
            min_oligos_per_gene=set_size_min,
        )
        check_content_oligo_database(oligo_database)

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(dir_database="1_db_probes_initial")
            print(f"Saved probe database for step 1 (Create Database) in directory {dir_database}")

        oligo_database = target_probe_designer.filter_by_property(
            oligo_database=oligo_database,
            GC_content_min=target_probe_GC_content_min,
            GC_content_max=target_probe_GC_content_max,
            Tm_min=target_probe_Tm_min,
            Tm_max=target_probe_Tm_max,
            detect_oligo_length_min=detection_oligo_length_min,
            detect_oligo_length_max=detection_oligo_length_max,
            min_thymines=detection_oligo_min_thymines,
            arm_length_min=target_probe_padlock_arm_length_min,
            arm_Tm_dif_max=target_probe_padlock_arm_Tm_dif_max,
            arm_Tm_min=target_probe_padlock_arm_Tm_min,
            arm_Tm_max=target_probe_padlock_arm_Tm_max,
            homopolymeric_base_n=target_probe_homopolymeric_base_n,
            Tm_parameters=self.target_probe_Tm_parameters,
            Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
        )
        check_content_oligo_database(oligo_database)

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(dir_database="2_db_probes_property_filter")
            print(f"Saved probe database for step 2 (Property Filters) in directory {dir_database}")

        oligo_database = target_probe_designer.filter_by_transcriptome_zero_hits(
            oligo_database=oligo_database,
            files_fasta_reference_database=files_fasta_reference_database_target_probe,
            specificity_blastn_search_parameters=self.target_probe_specificity_blastn_search_parameters,
            specificity_blastn_hit_parameters=self.target_probe_specificity_blastn_hit_parameters,
            ligation_region_size=target_probe_ligation_region_size,
            arm_length_min=target_probe_padlock_arm_length_min,
            arm_Tm_dif_max=target_probe_padlock_arm_Tm_dif_max,
            arm_Tm_min=target_probe_padlock_arm_Tm_min,
            arm_Tm_max=target_probe_padlock_arm_Tm_max,
            Tm_parameters=self.target_probe_Tm_parameters,
            Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
            Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
        )
        check_content_oligo_database(oligo_database)

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(dir_database="3_db_probes_specificity_filter")
            print(
                f"Saved probe database for step 3 (Transcriptome Zero-Hit Specificity Filters) in directory {dir_database}"
            )

        oligo_database = target_probe_designer.create_oligo_sets(
            oligo_database=oligo_database,
            isoform_weight=target_probe_isoform_weight,
            GC_content_min=target_probe_GC_content_min,
            GC_content_opt=target_probe_GC_content_opt,
            GC_content_max=target_probe_GC_content_max,
            GC_weight=target_probe_GC_weight,
            Tm_min=target_probe_Tm_min,
            Tm_opt=target_probe_Tm_opt,
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
        check_content_oligo_database(oligo_database)

        if self.write_intermediate_steps:
            dir_database = oligo_database.save_database(dir_database="4_db_probes_probesets")
            dir_probesets = oligo_database.write_oligosets_to_table()
            print(
                f"Saved probe database for step 4 (Set Selection) in directory {dir_database} and probeset table in directory {dir_probesets}"
            )

        return oligo_database


def _read_gene_ids(file_regions: str) -> Optional[list]:
    if file_regions is None:
        warnings.warn(
            "No region list file was provided! All entries from the custom FASTA file are used."
        )
        return None
    with open(file_regions) as handle:
        return list(set(line.rstrip() for line in handle.readlines()))


def _validate_single_sequence_fasta(files_fasta: list[str], allow_multiple: bool = False) -> None:
    if isinstance(files_fasta, str):
        files_fasta = [files_fasta]
    num_sequences = sum(1 for file_fasta in files_fasta for _ in SeqIO.parse(file_fasta, "fasta"))
    if num_sequences == 0:
        raise ValueError("files_fasta_target_probe_database does not contain any FASTA records.")
    if num_sequences > 1 and not allow_multiple:
        raise ValueError(
            "This workflow expects a single FASTA sequence. Set allow_multiple_target_sequences: true "
            "only if you intentionally want to design across multiple custom entries."
        )


def main():
    """Run the custom-sequence Scrinshot ISS pipeline."""
    print("--------------START PIPELINE--------------")

    args = base_parser()
    with open(args["config"], "r") as handle:
        config = yaml.safe_load(handle)

    _validate_single_sequence_fasta(
        files_fasta=config["files_fasta_target_probe_database"],
        allow_multiple=config.get("allow_multiple_target_sequences", False),
    )
    gene_ids = _read_gene_ids(config["file_regions"])

    pipeline = CustomSequenceScrinshotISSProbeDesigner(
        write_intermediate_steps=config["write_intermediate_steps"],
        dir_output=config["dir_output"],
        n_jobs=config["n_jobs"],
    )

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
        max_graph_size=config["max_graph_size"],
        n_attempts=config["n_attempts"],
        heuristic=config["heuristic"],
        heuristic_n_attempts=config["heuristic_n_attempts"],
        target_probe_Tm_parameters=config["target_probe_Tm_parameters"],
        target_probe_Tm_chem_correction_parameters=config[
            "target_probe_Tm_chem_correction_parameters"
        ],
        target_probe_Tm_salt_correction_parameters=config["target_probe_Tm_salt_correction_parameters"],
        detection_oligo_Tm_parameters=config["detection_oligo_Tm_parameters"],
        detection_oligo_Tm_chem_correction_parameters=config[
            "detection_oligo_Tm_chem_correction_parameters"
        ],
        detection_oligo_Tm_salt_correction_parameters=config[
            "detection_oligo_Tm_salt_correction_parameters"
        ],
    )

    backbone = config.get("padlock_backbone", {})
    pipeline.set_backbone_parameters(
        anchor_sequence=backbone.get("anchor_sequence", "TGCGTCTATTTAGTGGAGCC"),
        file_gene_to_lbar=backbone.get("file_gene_to_lbar"),
        file_lbar_to_sequence=backbone.get("file_lbar_to_sequence"),
        gene_column=backbone.get("gene_column", "Gene"),
        lbar_id_column_gene_table=backbone.get("lbar_id_column_gene_table", "Lbar_ID"),
        lbar_id_column_sequence_table=backbone.get("lbar_id_column_sequence_table", "Lbar_ID"),
        lbar_sequence_column=backbone.get("lbar_sequence_column", "Sequence"),
        gene_specific_sequence=backbone.get("gene_specific_sequence"),
        direct_lbar_id=backbone.get("direct_lbar_id", "custom"),
    )

    oligo_database = pipeline.design_target_probes(
        gene_ids=gene_ids,
        files_fasta_target_probe_database=config["files_fasta_target_probe_database"],
        files_fasta_reference_database_target_probe=config["files_fasta_reference_database_target_probe"],
        target_probe_length_min=config["target_probe_length_min"],
        target_probe_length_max=config["target_probe_length_max"],
        target_probe_isoform_weight=config.get("target_probe_isoform_weight", 0),
        target_probe_GC_content_min=config["target_probe_GC_content_min"],
        target_probe_GC_content_opt=config["target_probe_GC_content_opt"],
        target_probe_GC_content_max=config["target_probe_GC_content_max"],
        target_probe_GC_weight=config["target_probe_GC_weight"],
        target_probe_Tm_min=config["target_probe_Tm_min"],
        target_probe_Tm_opt=config["target_probe_Tm_opt"],
        target_probe_Tm_max=config["target_probe_Tm_max"],
        target_probe_Tm_weight=config["target_probe_Tm_weight"],
        target_probe_homopolymeric_base_n=config["target_probe_homopolymeric_base_n"],
        detection_oligo_min_thymines=config["detection_oligo_min_thymines"],
        detection_oligo_length_min=config["detection_oligo_length_min"],
        detection_oligo_length_max=config["detection_oligo_length_max"],
        target_probe_padlock_arm_length_min=config["target_probe_padlock_arm_length_min"],
        target_probe_padlock_arm_Tm_dif_max=config["target_probe_padlock_arm_Tm_dif_max"],
        target_probe_padlock_arm_Tm_min=config["target_probe_padlock_arm_Tm_min"],
        target_probe_padlock_arm_Tm_max=config["target_probe_padlock_arm_Tm_max"],
        target_probe_ligation_region_size=config["target_probe_ligation_region_size"],
        set_size_min=config["set_size_min"],
        set_size_opt=config["set_size_opt"],
        distance_between_target_probes=config["distance_between_target_probes"],
        n_sets=config["n_sets"],
    )

    flank_config = config.get("probe_flanks", {})
    if flank_config.get("enabled", False):
        oligo_database = pipeline.design_probe_flanks(
            oligo_database=oligo_database,
            files_fasta_target_context=flank_config.get(
                "files_fasta_target_context", config["files_fasta_target_probe_database"]
            ),
            flank_5prime_length=flank_config.get("flank_5prime_length", 0),
            flank_3prime_length=flank_config.get("flank_3prime_length", 0),
            flank_5prime_distance=flank_config.get("flank_5prime_distance", 0),
            flank_3prime_distance=flank_config.get("flank_3prime_distance", 0),
        )

    oligo_database = pipeline.design_padlock_backbone(oligo_database=oligo_database)
    pipeline.generate_output(oligo_database=oligo_database, top_n_sets=config["top_n_sets"])

    print("--------------END PIPELINE--------------")


if __name__ == "__main__":
    main()
