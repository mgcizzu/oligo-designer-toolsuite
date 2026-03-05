############################################
# imports
############################################

import warnings

import yaml

from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.pipelines._oligo_seq_probe_designer import OligoSeqProbeDesigner
from oligo_designer_toolsuite.pipelines._utils import base_parser


class FlexProbeDesigner(OligoSeqProbeDesigner):
    """
    FLEX probe designer for 10x GEX workflows on custom transcriptomes.

    This pipeline reuses the Oligo-Seq target probe design workflow and then
    builds FLEX constructs by concatenating configurable 5' and 3' handles
    around each designed target probe sequence.
    """

    def filter_by_ligation_junction(
        self,
        oligo_database: OligoDatabase,
        lhs_position: int = 25,
        required_lhs_base: str = "T",
        required_lhs_mode: str = "hard",
        prohibited_ligation_pairs: list[str] = None,
    ) -> OligoDatabase:
        """
        Filter probes by ligation-junction constraints.

        Junction is defined by probe bases at:
        - lhs_position (1-based; e.g. 25 for 10x recommendation)
        - lhs_position + 1
        """
        if prohibited_ligation_pairs is None:
            prohibited_ligation_pairs = []
        if lhs_position < 1:
            raise ValueError("lhs_position must be >= 1.")
        if required_lhs_mode not in {"hard", "soft", "off"}:
            raise ValueError("required_lhs_mode must be one of: 'hard', 'soft', 'off'.")

        required_lhs_base = required_lhs_base.upper() if required_lhs_base else None
        prohibited_ligation_pairs = [pair.upper() for pair in prohibited_ligation_pairs]

        removed_for_short_length = 0
        removed_for_required_base = 0
        removed_for_prohibited_pair = 0
        new_oligo_attributes = {}

        for region_id in list(oligo_database.database.keys()):
            for oligo_id in list(oligo_database.database[region_id].keys()):
                probe_sequence = oligo_database.get_oligo_attribute_value(
                    attribute="oligo", region_id=region_id, oligo_id=oligo_id, flatten=True
                )
                probe_sequence = str(probe_sequence).upper()

                if len(probe_sequence) < lhs_position + 1:
                    del oligo_database.database[region_id][oligo_id]
                    removed_for_short_length += 1
                    continue

                lhs_base = probe_sequence[lhs_position - 1]
                rhs_base = probe_sequence[lhs_position]
                ligation_pair = lhs_base + rhs_base

                passes_required_lhs = True
                if required_lhs_mode != "off" and required_lhs_base:
                    passes_required_lhs = lhs_base == required_lhs_base
                    if required_lhs_mode == "hard" and not passes_required_lhs:
                        del oligo_database.database[region_id][oligo_id]
                        removed_for_required_base += 1
                        continue

                if ligation_pair in prohibited_ligation_pairs:
                    del oligo_database.database[region_id][oligo_id]
                    removed_for_prohibited_pair += 1
                    continue

                new_oligo_attributes[oligo_id] = {
                    "ligation_lhs_position": lhs_position,
                    "ligation_lhs_base": lhs_base,
                    "ligation_rhs_base": rhs_base,
                    "ligation_probe_pair": ligation_pair,
                    "ligation_required_lhs_mode": required_lhs_mode,
                    "ligation_required_lhs_base": required_lhs_base,
                    "ligation_required_lhs_pass": passes_required_lhs,
                }

        oligo_database.update_oligo_attributes(new_oligo_attributes)
        oligo_database.remove_regions_with_insufficient_oligos(
            pipeline_step="FLEX Ligation Junction Filter"
        )

        removed_total = (
            removed_for_short_length + removed_for_required_base + removed_for_prohibited_pair
        )
        if removed_total > 0:
            warnings.warn(
                "FLEX ligation filter removed "
                f"{removed_total} probes "
                f"(short={removed_for_short_length}, required_lhs={removed_for_required_base}, prohibited_pair={removed_for_prohibited_pair})."
            )

        return oligo_database

    def design_flex_probes(
        self,
        oligo_database: OligoDatabase,
        handle_5prime: str,
        handle_3prime: str,
        linker: str = "",
    ) -> OligoDatabase:
        """Attach FLEX assay sequences to each designed target probe."""
        new_oligo_attributes = {}

        for region_id, database_region in oligo_database.database.items():
            for oligo_id, _ in database_region.items():
                sequence_target_probe = oligo_database.get_oligo_attribute_value(
                    attribute="oligo", region_id=region_id, oligo_id=oligo_id, flatten=True
                )
                sequence_flex_probe = f"{handle_5prime}{linker}{sequence_target_probe}{handle_3prime}"

                new_oligo_attributes[oligo_id] = {
                    "sequence_target_probe": sequence_target_probe,
                    "sequence_flex_probe": sequence_flex_probe,
                    "sequence_handle_5prime": handle_5prime,
                    "sequence_linker": linker,
                    "sequence_handle_3prime": handle_3prime,
                }

        oligo_database.update_oligo_attributes(new_oligo_attributes)
        return oligo_database

    def generate_output(
        self,
        oligo_database: OligoDatabase,
        top_n_sets: int = 3,
        attributes: list = None,
    ) -> None:
        """Generate output with FLEX-specific sequence attributes."""
        if attributes is None:
            attributes = [
                "source",
                "species",
                "annotation_release",
                "genome_assembly",
                "regiontype",
                "gene_id",
                "transcript_id",
                "exon_number",
                "chromosome",
                "start",
                "end",
                "strand",
                "oligo",
                "target",
                "sequence_target_probe",
                "sequence_flex_probe",
                "sequence_handle_5prime",
                "sequence_linker",
                "sequence_handle_3prime",
                "ligation_lhs_position",
                "ligation_lhs_base",
                "ligation_rhs_base",
                "ligation_probe_pair",
                "ligation_required_lhs_mode",
                "ligation_required_lhs_base",
                "ligation_required_lhs_pass",
                "length",
                "GC_content",
                "TmNN",
                "num_targeted_transcripts",
                "number_total_transcripts",
                "isoform_consensus",
                "length_selfcomplement",
                "DG_secondary_structure",
            ]

        super().generate_output(
            oligo_database=oligo_database,
            top_n_sets=top_n_sets,
            attributes=attributes,
        )


def main():
    """Run FLEX probe designer pipeline."""
    print("--------------START PIPELINE--------------")

    args = base_parser()

    with open(args["config"], "r") as handle:
        config = yaml.safe_load(handle)

    if config["file_regions"] is None:
        warnings.warn(
            "No gene list file was provided! All genes from fasta file are used to generate the probes."
        )
        gene_ids = None
    else:
        with open(config["file_regions"]) as handle:
            lines = handle.readlines()
            gene_ids = list(set([line.rstrip() for line in lines]))

    pipeline = FlexProbeDesigner(
        write_intermediate_steps=config["write_intermediate_steps"],
        dir_output=config["dir_output"],
        n_jobs=config["n_jobs"],
    )

    pipeline.set_developer_parameters(
        target_probe_hybridization_probability_alignment_method=config[
            "target_probe_hybridization_probability_alignment_method"
        ],
        target_probe_hybridization_probability_blastn_search_parameters=config[
            "target_probe_hybridization_probability_blastn_search_parameters"
        ],
        target_probe_hybridization_probability_blastn_hit_parameters=config[
            "target_probe_hybridization_probability_blastn_hit_parameters"
        ],
        target_probe_hybridization_probability_bowtie_search_parameters=config[
            "target_probe_hybridization_probability_bowtie_search_parameters"
        ],
        target_probe_hybridization_probability_bowtie_hit_parameters=config[
            "target_probe_hybridization_probability_bowtie_hit_parameters"
        ],
        target_probe_cross_hybridization_alignment_method=config[
            "target_probe_cross_hybridization_alignment_method"
        ],
        target_probe_cross_hybridization_blastn_search_parameters=config[
            "target_probe_cross_hybridization_blastn_search_parameters"
        ],
        target_probe_cross_hybridization_blastn_hit_parameters=config[
            "target_probe_cross_hybridization_blastn_hit_parameters"
        ],
        target_probe_cross_hybridization_bowtie_search_parameters=config[
            "target_probe_cross_hybridization_bowtie_search_parameters"
        ],
        target_probe_cross_hybridization_bowtie_hit_parameters=config[
            "target_probe_cross_hybridization_bowtie_hit_parameters"
        ],
        max_graph_size=config["max_graph_size"],
        n_attempts=config["n_attempts"],
        heuristic=config["heuristic"],
        heuristic_n_attempts=config["heuristic_n_attempts"],
        target_probe_Tm_parameters=config["target_probe_Tm_parameters"],
        target_probe_Tm_chem_correction_parameters=config["target_probe_Tm_chem_correction_parameters"],
        target_probe_Tm_salt_correction_parameters=config["target_probe_Tm_salt_correction_parameters"],
    )

    oligo_database = pipeline.design_target_probes(
        files_fasta_target_probe_database=config["files_fasta_target_probe_database"],
        files_fasta_reference_database_target_probe=config["files_fasta_reference_database_target_probe"],
        gene_ids=gene_ids,
        target_probe_length_min=config["target_probe_length_min"],
        target_probe_length_max=config["target_probe_length_max"],
        target_probe_split_region=config["target_probe_split_region"],
        target_probe_targeted_exons=config["target_probe_targeted_exons"],
        target_probe_isoform_consensus=config["target_probe_isoform_consensus"],
        target_probe_GC_content_min=config["target_probe_GC_content_min"],
        target_probe_GC_content_opt=config["target_probe_GC_content_opt"],
        target_probe_GC_content_max=config["target_probe_GC_content_max"],
        target_probe_Tm_min=config["target_probe_Tm_min"],
        target_probe_Tm_opt=config["target_probe_Tm_opt"],
        target_probe_Tm_max=config["target_probe_Tm_max"],
        target_probe_secondary_structures_T=config["target_probe_secondary_structures_T"],
        target_probe_secondary_structures_threshold_deltaG=config[
            "target_probe_secondary_structures_threshold_deltaG"
        ],
        target_probe_homopolymeric_base_n=config["target_probe_homopolymeric_base_n"],
        target_probe_max_len_selfcomplement=config["target_probe_max_len_selfcomplement"],
        target_probe_hybridization_probability_threshold=config[
            "target_probe_hybridization_probability_threshold"
        ],
        target_probe_apply_cross_hybridization=config.get(
            "target_probe_apply_cross_hybridization", True
        ),
        target_probe_GC_weight=config["target_probe_GC_weight"],
        target_probe_Tm_weight=config["target_probe_Tm_weight"],
        set_size_min=config["set_size_min"],
        set_size_opt=config["set_size_opt"],
        distance_between_target_probes=config["distance_between_target_probes"],
        n_sets=config["n_sets"],
    )

    flex_params = config.get("flex_probe", {})
    ligation_params = flex_params.get("ligation_filter", {})
    if ligation_params.get("enabled", True):
        oligo_database = pipeline.filter_by_ligation_junction(
            oligo_database=oligo_database,
            lhs_position=ligation_params.get("lhs_position", 25),
            required_lhs_base=ligation_params.get("required_lhs_base", "T"),
            required_lhs_mode=ligation_params.get("required_lhs_mode", "hard"),
            prohibited_ligation_pairs=ligation_params.get("prohibited_ligation_pairs", []),
        )

    oligo_database = pipeline.design_flex_probes(
        oligo_database=oligo_database,
        handle_5prime=flex_params.get("handle_5prime", ""),
        handle_3prime=flex_params.get("handle_3prime", ""),
        linker=flex_params.get("linker", ""),
    )

    pipeline.generate_output(oligo_database=oligo_database, top_n_sets=config["top_n_sets"])

    print("--------------END PIPELINE--------------")


if __name__ == "__main__":
    main()
