############################################
# imports
############################################

import re
import warnings

import pandas as pd
import yaml
from Bio import SeqIO

from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.pipelines._scrinshot_probe_designer import ScrinshotProbeDesigner
from oligo_designer_toolsuite.pipelines._utils import base_parser
from oligo_designer_toolsuite.utils import FastaParser


class ScrinshotISSProbeDesigner(ScrinshotProbeDesigner):
    """
    ISS variant of ``ScrinshotProbeDesigner`` with a configurable 2-part backbone.

    The full target-probe and detection-oligo workflow is inherited from the
    base Scrinshot pipeline. This variant only overrides padlock backbone
    construction so users can use:
    1) a constant anchor sequence
    2) a gene-specific sequence loaded through two CSV mapping tables.
    """

    def __init__(self, write_intermediate_steps: bool, dir_output: str, n_jobs: int) -> None:
        super().__init__(
            write_intermediate_steps=write_intermediate_steps,
            dir_output=dir_output,
            n_jobs=n_jobs,
        )
        self.set_backbone_parameters()

    def set_backbone_parameters(
        self,
        anchor_sequence: str = "TGCGTCTATTTAGTGGAGCC",
        file_gene_to_lbar: str = None,
        file_lbar_to_sequence: str = None,
        gene_column: str = "Gene",
        lbar_id_column_gene_table: str = "Lbar_ID",
        lbar_id_column_sequence_table: str = "Lbar_ID",
        lbar_sequence_column: str = "Sequence",
    ) -> None:
        """Configure anchor and CSV-based gene-specific backbone mapping."""

        if not anchor_sequence:
            raise ValueError("anchor_sequence must be a non-empty DNA sequence")
        if re.fullmatch(r"[ACGT]+", anchor_sequence.upper()) is None:
            raise ValueError("anchor_sequence must contain only A/C/G/T")

        self.anchor_sequence = anchor_sequence.upper()
        self.file_gene_to_lbar = file_gene_to_lbar
        self.file_lbar_to_sequence = file_lbar_to_sequence
        self.gene_column = gene_column
        self.lbar_id_column_gene_table = lbar_id_column_gene_table
        self.lbar_id_column_sequence_table = lbar_id_column_sequence_table
        self.lbar_sequence_column = lbar_sequence_column

    def _load_gene_specific_backbone_sequences(self) -> dict[str, tuple[str, str]]:
        """
        Load and validate mappings:
        Gene -> Lbar_ID and Lbar_ID -> Sequence.

        Returns:
            dict[gene] = (lbar_id, sequence)
        """
        if not self.file_gene_to_lbar or not self.file_lbar_to_sequence:
            raise ValueError(
                "Both file_gene_to_lbar and file_lbar_to_sequence must be provided in padlock_backbone config."
            )

        table_gene_to_lbar = pd.read_csv(self.file_gene_to_lbar)
        table_lbar_to_sequence = pd.read_csv(self.file_lbar_to_sequence)

        required_gene_cols = {self.gene_column, self.lbar_id_column_gene_table}
        missing_gene_cols = required_gene_cols - set(table_gene_to_lbar.columns)
        if missing_gene_cols:
            raise ValueError(
                f"Missing columns in Gene->Lbar_ID table {self.file_gene_to_lbar}: {sorted(missing_gene_cols)}"
            )

        required_lbar_cols = {self.lbar_id_column_sequence_table, self.lbar_sequence_column}
        missing_lbar_cols = required_lbar_cols - set(table_lbar_to_sequence.columns)
        if missing_lbar_cols:
            raise ValueError(
                f"Missing columns in Lbar_ID->Sequence table {self.file_lbar_to_sequence}: {sorted(missing_lbar_cols)}"
            )

        table_gene_to_lbar[self.gene_column] = (
            table_gene_to_lbar[self.gene_column].astype(str).str.strip()
        )
        table_gene_to_lbar[self.lbar_id_column_gene_table] = (
            table_gene_to_lbar[self.lbar_id_column_gene_table].astype(str).str.strip()
        )
        table_lbar_to_sequence[self.lbar_id_column_sequence_table] = (
            table_lbar_to_sequence[self.lbar_id_column_sequence_table].astype(str).str.strip()
        )
        table_lbar_to_sequence[self.lbar_sequence_column] = (
            table_lbar_to_sequence[self.lbar_sequence_column].astype(str).str.strip().str.upper()
        )

        if table_gene_to_lbar[self.gene_column].duplicated().any():
            duplicated_genes = sorted(
                table_gene_to_lbar.loc[
                    table_gene_to_lbar[self.gene_column].duplicated(), self.gene_column
                ].unique()
            )
            raise ValueError(
                f"Gene column '{self.gene_column}' must be unique in {self.file_gene_to_lbar}. "
                f"Duplicated genes: {duplicated_genes}"
            )

        if table_lbar_to_sequence[self.lbar_id_column_sequence_table].duplicated().any():
            duplicated_lbar_ids = sorted(
                table_lbar_to_sequence.loc[
                    table_lbar_to_sequence[self.lbar_id_column_sequence_table].duplicated(),
                    self.lbar_id_column_sequence_table,
                ].unique()
            )
            raise ValueError(
                f"Lbar_ID column '{self.lbar_id_column_sequence_table}' must be unique in {self.file_lbar_to_sequence}. "
                f"Duplicated IDs: {duplicated_lbar_ids}"
            )

        lbar_to_sequence = dict(
            zip(
                table_lbar_to_sequence[self.lbar_id_column_sequence_table],
                table_lbar_to_sequence[self.lbar_sequence_column],
            )
        )

        gene_to_lbar_sequence = {}
        for _, row in table_gene_to_lbar.iterrows():
            gene = row[self.gene_column]
            lbar_id = row[self.lbar_id_column_gene_table]
            if lbar_id not in lbar_to_sequence:
                raise ValueError(
                    f"Lbar_ID '{lbar_id}' for gene '{gene}' is missing from {self.file_lbar_to_sequence}."
                )
            sequence = lbar_to_sequence[lbar_id]
            if re.fullmatch(r"[ACGT]+", sequence) is None:
                raise ValueError(
                    f"Invalid DNA sequence for Lbar_ID '{lbar_id}' in {self.file_lbar_to_sequence}: '{sequence}'"
                )
            gene_to_lbar_sequence[gene] = (lbar_id, sequence)

        return gene_to_lbar_sequence

    def design_padlock_backbone(self, oligo_database: OligoDatabase) -> OligoDatabase:
        """
        Design backbones as: anchor_sequence + gene_specific_sequence.

        The gene-specific sequence is derived from:
        Gene -> Lbar_ID and Lbar_ID -> Sequence tables.
        """
        gene_to_lbar_sequence = self._load_gene_specific_backbone_sequences()
        region_ids = list(oligo_database.database.keys())

        for region_id in region_ids:
            if region_id not in gene_to_lbar_sequence:
                raise ValueError(
                    f"Region/Gene '{region_id}' not found in Gene column '{self.gene_column}' "
                    f"of {self.file_gene_to_lbar}."
                )
            lbar_id, sequence_gene_specific = gene_to_lbar_sequence[region_id]

            oligo_sets_region = oligo_database.oligosets[region_id]
            oligoset_oligo_columns = [
                col for col in oligo_sets_region.columns if col.startswith("oligo_")
            ]

            new_oligo_attributes = {}
            for index in range(len(oligo_sets_region.index)):
                for column in oligoset_oligo_columns:
                    oligo_id = str(oligo_sets_region.loc[index, column])

                    ligation_site = oligo_database.get_oligo_attribute_value(
                        attribute="ligation_site", region_id=region_id, oligo_id=oligo_id, flatten=True
                    )
                    sequence_oligo = oligo_database.get_oligo_attribute_value(
                        attribute="oligo", region_id=region_id, oligo_id=oligo_id, flatten=True
                    )

                    sequence_padlock_arm1 = sequence_oligo[ligation_site:]
                    sequence_padlock_arm2 = sequence_oligo[:ligation_site]
                    sequence_padlock_backbone = self.anchor_sequence + sequence_gene_specific
                    sequence_padlock_probe = (
                        sequence_padlock_arm1 + sequence_padlock_backbone + sequence_padlock_arm2
                    )

                    Tm_arm1 = self.oligo_attributes_calculator._calc_TmNN(
                        sequence=sequence_padlock_arm1,
                        Tm_parameters=self.target_probe_Tm_parameters,
                        Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
                        Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
                    )
                    Tm_arm2 = self.oligo_attributes_calculator._calc_TmNN(
                        sequence=sequence_padlock_arm2,
                        Tm_parameters=self.target_probe_Tm_parameters,
                        Tm_chem_correction_parameters=self.target_probe_Tm_chem_correction_parameters,
                        Tm_salt_correction_parameters=self.target_probe_Tm_salt_correction_parameters,
                    )

                    new_oligo_attributes[oligo_id] = {
                        "lbar_id": lbar_id,
                        "sequence_gene_specific": sequence_gene_specific,
                        "sequence_padlock_anchor": self.anchor_sequence,
                        # Keep these legacy fields populated for compatibility with inherited output schema.
                        "barcode": sequence_gene_specific,
                        "sequence_target": oligo_database.get_oligo_attribute_value(
                            attribute="target", region_id=region_id, oligo_id=oligo_id, flatten=True
                        ),
                        "sequence_target_probe": oligo_database.get_oligo_attribute_value(
                            attribute="oligo", region_id=region_id, oligo_id=oligo_id, flatten=True
                        ),
                        "sequence_padlock_arm1": sequence_padlock_arm1,
                        "sequence_padlock_arm2": sequence_padlock_arm2,
                        "sequence_padlock_accessory1": self.anchor_sequence,
                        "sequence_padlock_ISS_anchor": self.anchor_sequence,
                        "sequence_padlock_accessory2": "",
                        "sequence_padlock_backbone": sequence_padlock_backbone,
                        "sequence_padlock_probe": sequence_padlock_probe,
                        "Tm_arm1": Tm_arm1,
                        "Tm_arm2": Tm_arm2,
                        "Tm_diff_arms": round(abs(Tm_arm1 - Tm_arm2), 2),
                    }

            oligo_database.update_oligo_attributes(new_oligo_attributes)

        return oligo_database

    def _build_target_context_index(self, files_fasta_target_context: list[str]) -> dict:
        """
        Build per-region sequence contexts in transcript orientation.

        For each FASTA entry, we derive a coordinate map of the same length as the
        sequence, where each nucleotide is associated with a genomic coordinate.
        """
        fasta_parser = FastaParser()
        context_index = {}

        for file_fasta in files_fasta_target_context:
            for entry in SeqIO.parse(file_fasta, "fasta"):
                region_id, _, coordinates = fasta_parser.parse_fasta_header(
                    header=entry.id, parse_additional_info=False
                )
                chromosome = coordinates["chromosome"][0]
                strand = coordinates["strand"][0]

                # We can only compute flanks when coordinates are available.
                if chromosome is None or strand is None:
                    continue

                coordinate_map = []
                for start, end in zip(coordinates["start"], coordinates["end"]):
                    coordinate_map.extend(range(start, end + 1))

                # Keep 5'->3' transcript orientation, matching sequence generation behavior.
                if strand == "-":
                    coordinate_map.reverse()

                sequence = str(entry.seq).upper()
                if len(sequence) != len(coordinate_map):
                    warnings.warn(
                        f"Skipping context entry '{entry.id}' because sequence and coordinate lengths differ."
                    )
                    continue

                context_index.setdefault(region_id, []).append(
                    {
                        "sequence": sequence,
                        "coordinate_map": coordinate_map,
                        "strand": strand,
                        "header": entry.id,
                    }
                )

        return context_index

    def _get_target_location_in_context(
        self,
        region_contexts: list[dict],
        target_sequence: str,
        start: int,
        end: int,
        strand: str,
    ) -> tuple:
        """Locate target sequence in region contexts and return (context, index_start)."""
        target_length = len(target_sequence)

        # Primary matching path: sequence + coordinates.
        for context in region_contexts:
            if context["strand"] != strand:
                continue
            sequence = context["sequence"]
            coordinate_map = context["coordinate_map"]

            idx = sequence.find(target_sequence)
            while idx != -1:
                if idx + target_length <= len(coordinate_map):
                    if strand == "+":
                        coordinates_match = (
                            coordinate_map[idx] == start and coordinate_map[idx + target_length - 1] == end
                        )
                    else:
                        coordinates_match = (
                            coordinate_map[idx] == end and coordinate_map[idx + target_length - 1] == start
                        )
                    if coordinates_match:
                        return context, idx
                idx = sequence.find(target_sequence, idx + 1)

        # Fallback: coordinate-only if uniquely identifiable.
        candidates = []
        for context in region_contexts:
            if context["strand"] != strand:
                continue
            coordinate_map = context["coordinate_map"]
            for idx in range(len(coordinate_map) - target_length + 1):
                if strand == "+":
                    coordinates_match = (
                        coordinate_map[idx] == start and coordinate_map[idx + target_length - 1] == end
                    )
                else:
                    coordinates_match = (
                        coordinate_map[idx] == end and coordinate_map[idx + target_length - 1] == start
                    )
                if coordinates_match:
                    candidates.append((context, idx))

        if len(candidates) == 1:
            return candidates[0]

        return None, None

    def design_probe_flanks(
        self,
        oligo_database: OligoDatabase,
        files_fasta_target_context: list[str],
        flank_5prime_length: int,
        flank_3prime_length: int,
        flank_5prime_distance: int = 0,
        flank_3prime_distance: int = 0,
    ) -> OligoDatabase:
        """
        Add 5' and 3' flanking sequences relative to transcript strand.

        5' flank:
            starts ``flank_5prime_distance`` nt upstream of target start and has
            length ``flank_5prime_length``.
        3' flank:
            starts ``flank_3prime_distance`` nt downstream of target end and has
            length ``flank_3prime_length``.
        """
        for value_name, value in {
            "flank_5prime_length": flank_5prime_length,
            "flank_3prime_length": flank_3prime_length,
            "flank_5prime_distance": flank_5prime_distance,
            "flank_3prime_distance": flank_3prime_distance,
        }.items():
            if value < 0:
                raise ValueError(f"{value_name} must be >= 0.")

        context_index = self._build_target_context_index(files_fasta_target_context)
        new_oligo_attributes = {}
        missing_context = 0
        out_of_bounds = 0

        for region_id in oligo_database.database.keys():
            region_contexts = context_index.get(region_id, [])
            for oligo_id in oligo_database.database[region_id].keys():
                target_sequence = oligo_database.get_oligo_attribute_value(
                    attribute="target", region_id=region_id, oligo_id=oligo_id, flatten=True
                )
                start = oligo_database.get_oligo_attribute_value(
                    attribute="start", region_id=region_id, oligo_id=oligo_id, flatten=True
                )
                end = oligo_database.get_oligo_attribute_value(
                    attribute="end", region_id=region_id, oligo_id=oligo_id, flatten=True
                )
                strand = oligo_database.get_oligo_attribute_value(
                    attribute="strand", region_id=region_id, oligo_id=oligo_id, flatten=True
                )

                # Normalize flattened values that may still be list-like.
                if isinstance(start, list):
                    start = min(start)
                if isinstance(end, list):
                    end = max(end)
                if isinstance(strand, list):
                    strand = strand[0]
                target_sequence = str(target_sequence).upper()

                context, target_idx = self._get_target_location_in_context(
                    region_contexts=region_contexts,
                    target_sequence=target_sequence,
                    start=int(start),
                    end=int(end),
                    strand=str(strand),
                )
                if context is None:
                    missing_context += 1
                    flank_5prime = None
                    flank_3prime = None
                else:
                    sequence_context = context["sequence"]
                    target_len = len(target_sequence)

                    left_start = target_idx - flank_5prime_distance - flank_5prime_length
                    left_end = target_idx - flank_5prime_distance
                    right_start = target_idx + target_len + flank_3prime_distance
                    right_end = right_start + flank_3prime_length

                    flank_5prime = None
                    flank_3prime = None

                    if flank_5prime_length == 0:
                        flank_5prime = ""
                    elif left_start >= 0 and left_end <= len(sequence_context):
                        flank_5prime = sequence_context[left_start:left_end]
                    else:
                        out_of_bounds += 1

                    if flank_3prime_length == 0:
                        flank_3prime = ""
                    elif right_start >= 0 and right_end <= len(sequence_context):
                        flank_3prime = sequence_context[right_start:right_end]
                    else:
                        out_of_bounds += 1

                new_oligo_attributes[oligo_id] = {
                    "sequence_flank_5prime": flank_5prime,
                    "sequence_flank_3prime": flank_3prime,
                }

        oligo_database.update_oligo_attributes(new_oligo_attributes)

        if missing_context > 0:
            warnings.warn(
                f"Could not resolve transcript-context location for {missing_context} probes; flank sequences were set to None."
            )
        if out_of_bounds > 0:
            warnings.warn(
                f"{out_of_bounds} requested flanks were out of context bounds; corresponding flank sequences were set to None."
            )

        return oligo_database

    def generate_output(
        self,
        oligo_database: OligoDatabase,
        top_n_sets: int = 3,
        attributes: list = None,
    ) -> None:
        """Generate final output and include ISS-specific attributes by default."""
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
                "sequence_padlock_probe",
                "sequence_detection_oligo",
                "sequence_padlock_arm1",
                "sequence_padlock_accessory1",
                "sequence_padlock_ISS_anchor",
                "barcode",
                "sequence_padlock_accessory2",
                "sequence_padlock_arm2",
                "sequence_target",
                "sequence_target_probe",
                "length",
                "ligation_site",
                "Tm_arm1",
                "Tm_arm2",
                "Tm_diff_arms",
                "Tm_detection_oligo",
                "isoform_consensus",
                "lbar_id",
                "sequence_gene_specific",
                "sequence_padlock_anchor",
                "sequence_flank_5prime",
                "sequence_flank_3prime",
            ]

        super().generate_output(
            oligo_database=oligo_database,
            top_n_sets=top_n_sets,
            attributes=attributes,
        )


def main():
    """Run the Scrinshot ISS pipeline with CSV-driven gene-specific backbone."""
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
            gene_ids = list(set(line.rstrip() for line in handle.readlines()))

    pipeline = ScrinshotISSProbeDesigner(
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
        target_probe_Tm_chem_correction_parameters=config["target_probe_Tm_chem_correction_parameters"],
        target_probe_Tm_salt_correction_parameters=config["target_probe_Tm_salt_correction_parameters"],
        detection_oligo_Tm_parameters=config["detection_oligo_Tm_parameters"],
        detection_oligo_Tm_chem_correction_parameters=config["detection_oligo_Tm_chem_correction_parameters"],
        detection_oligo_Tm_salt_correction_parameters=config["detection_oligo_Tm_salt_correction_parameters"],
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
    )

    oligo_database = pipeline.design_target_probes(
        gene_ids=gene_ids,
        files_fasta_target_probe_database=config["files_fasta_target_probe_database"],
        files_fasta_reference_database_target_probe=config["files_fasta_reference_database_target_probe"],
        target_probe_length_min=config["target_probe_length_min"],
        target_probe_length_max=config["target_probe_length_max"],
        target_probe_isoform_consensus=config["target_probe_isoform_consensus"],
        target_probe_isoform_weight=config["target_probe_isoform_weight"],
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

    oligo_database = pipeline.design_detection_oligos(
        oligo_database=oligo_database,
        detection_oligo_length_min=config["detection_oligo_length_min"],
        detection_oligo_length_max=config["detection_oligo_length_max"],
        detection_oligo_min_thymines=config["detection_oligo_min_thymines"],
        detection_oligo_U_distance=config["detection_oligo_U_distance"],
        detection_oligo_Tm_opt=config["detection_oligo_Tm_opt"],
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
