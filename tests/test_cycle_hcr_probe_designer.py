import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml
from openpyxl import Workbook


SCRIPT_CYCLE_HCR_PROBE_DESIGNER = "oligo_designer_toolsuite/pipelines/_cycle_hcr_probe_designer.py"


class TestCycleHCRProbeDesigner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cycle_hcr_test_"))
        self.fasta_path = self.tmpdir / "transcriptome.fna"
        self.genes_path = self.tmpdir / "genes.txt"
        self.assignment_path = self.tmpdir / "assignment.csv"
        self.workbook_path = self.tmpdir / "barcodes.xlsx"
        self.config_path = self.tmpdir / "config.yaml"
        self.output_dir = self.tmpdir / "output_cycle_hcr"

        left_target = "AGCGTCGATCGGCTAGCGATCGTAGCTAGCGTACGATCGTAGCTA"
        right_target = "TCGATCGTAGCGATGCTAGCTCGATCGTAGCTAGCGATCGTAGCA"
        filler = "GATCGATCGTAGCTAGCGATGCTAGCTAG"
        self.target_sequence = left_target + "TT" + right_target + filler

        with self.fasta_path.open("w") as handle:
            handle.write(">Arc::synthetic\n")
            handle.write(self.target_sequence + "\n")

        with self.genes_path.open("w") as handle:
            handle.write("Arc\n")

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "hcr488_B4"
        sheet.append(["P488L_1", "CCTCAACCTACCTCCAACAATCATCTCAGTTAGT", "P488R_7", "ACAGTTCTACGAATAATCTCACCATATTCgCTTC"])
        workbook.save(self.workbook_path)

        with self.assignment_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["Gene", "Sheet", "Left_Barcode_Name", "Right_Barcode_Name"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Gene": "Arc",
                    "Sheet": "hcr488_B4",
                    "Left_Barcode_Name": "P488L_1",
                    "Right_Barcode_Name": "P488R_7",
                }
            )

        config = {
            "n_jobs": 1,
            "dir_output": str(self.output_dir),
            "write_intermediate_steps": True,
            "top_n_sets": 1,
            "file_regions": str(self.genes_path),
            "files_fasta_target_probe_database": [str(self.fasta_path)],
            "files_fasta_reference_database_target_probe": [str(self.fasta_path)],
            "barcode_library": {
                "file": str(self.workbook_path),
                "sheets": ["hcr488_B4"],
            },
            "barcode_assignment": {
                "file": str(self.assignment_path),
                "gene_column": "Gene",
                "sheet_column": "Sheet",
                "left_barcode_name_column": "Left_Barcode_Name",
                "right_barcode_name_column": "Right_Barcode_Name",
                "require_unique_barcode_combination": True,
                "require_same_sheet_for_left_right": True,
            },
            "target_probe_length_min": 45,
            "target_probe_length_max": 45,
            "target_probe_isoform_consensus": 0,
            "target_probe_GC_content_min": 30,
            "target_probe_GC_content_opt": 60,
            "target_probe_GC_content_max": 90,
            "target_probe_Tm_min": 0,
            "target_probe_Tm_opt": 72,
            "target_probe_Tm_max": 76,
            "target_probe_homopolymeric_base_n": {"A": 6, "T": 6, "C": 6, "G": 6},
            "target_probe_T_secondary_structure": 76,
            "target_probe_secondary_structures_threshold_deltaG": 0,
            "target_probe_apply_cross_hybridization": True,
            "target_probe_GC_weight": 1,
            "target_probe_Tm_weight": 1,
            "target_probe_isoform_weight": 1,
            "set_size_min": 1,
            "set_size_opt": 1,
            "distance_between_target_probes": 5,
            "n_sets": 1,
            "cycle_hcr": {
                "selection_mode": "sliding_window",
                "window_length": 92,
                "left_probe_length": 45,
                "inter_probe_gap_length": 2,
                "right_probe_length": 45,
                "max_secondary_structure_tm": 76,
                "max_cross_hybridization_tm": 72,
                "enable_cross_hybridization_tm_screen": False,
                "cross_hybridization_seed_length": 12,
                "min_dna_rna_tm_estimate": 90,
                "dna_rna_tm_estimate_offset": 10,
                "junction_length": 26,
                "junction_reference_scope": "transcriptome",
                "junction_max_reference_matches": 1,
            },
            "output_modes": ["direct", "twist_pcr_t7_rt"],
            "direct_primary": {
                "spacer": "TT",
                "left_probe_order": [
                    "target_left_reverse_complement",
                    "spacer",
                    "left_barcode_full",
                ],
                "right_probe_order": [
                    "right_barcode_full",
                    "spacer",
                    "target_right_reverse_complement",
                ],
            },
            "twist_pcr_t7_rt": {
                "forward_primer_with_t7": "TAATACGACTCACTATAGCGTCATC",
                "reverse_primer_sequence": "CGACACCGAACGTGCGACAA",
                "spacer": "TT",
                "left_barcode_subsequence": {"source": "left_barcode", "side": "terminal", "length": 14},
                "right_barcode_subsequence": {"source": "right_barcode", "side": "leading", "length": 14},
                "left_probe_order": [
                    "forward_primer_with_t7",
                    "target_left_sense",
                    "spacer",
                    "left_barcode_subsequence",
                    "reverse_primer_sequence",
                ],
                "right_probe_order": [
                    "forward_primer_with_t7",
                    "right_barcode_subsequence",
                    "spacer",
                    "target_right_sense",
                    "reverse_primer_sequence",
                ],
            },
            "target_probe_specificity_blastn_search_parameters": {
                "perc_identity": 80,
                "strand": "minus",
                "word_size": 10,
                "dust": "no",
                "soft_masking": "false",
                "max_target_seqs": 10,
                "max_hsps": 1000,
            },
            "target_probe_specificity_blastn_hit_parameters": {"min_alignment_length": 17},
            "target_probe_cross_hybridization_blastn_search_parameters": {
                "perc_identity": 80,
                "strand": "minus",
                "word_size": 7,
                "dust": "no",
                "soft_masking": "false",
                "max_target_seqs": 10,
            },
            "target_probe_cross_hybridization_blastn_hit_parameters": {"min_alignment_length": 17},
            "target_probe_Tm_parameters": {
                "nn_table": "DNA_NN3",
                "tmm_table": "DNA_TMM1",
                "imm_table": "DNA_IMM1",
                "de_table": "DNA_DE1",
                "dnac1": 50,
                "dnac2": 0,
                "saltcorr": 7,
                "Na": 1000,
                "K": 0,
                "Tris": 0,
                "Mg": 0,
                "dNTPs": 0,
            },
            "target_probe_Tm_chem_correction_parameters": None,
            "target_probe_Tm_salt_correction_parameters": None,
            "max_graph_size": 5000,
            "n_attempts": 100000,
            "heuristic": True,
            "heuristic_n_attempts": 100,
        }
        with self.config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def _run_pipeline(self):
        return subprocess.run(
            [
                sys.executable,
                os.path.abspath(SCRIPT_CYCLE_HCR_PROBE_DESIGNER),
                f"-c{self.config_path}",
            ],
            capture_output=True,
            text=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_cycle_hcr_pipeline_generates_direct_and_twist_outputs(self):
        result = self._run_pipeline()
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        panel = pd.read_table(self.output_dir / "cycle_hcr_probe_panel.tsv")
        pairs = pd.read_table(self.output_dir / "cycle_hcr_probe_pairs.tsv")
        assignments = pd.read_table(self.output_dir / "cycle_hcr_barcode_assignments.tsv")

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments.iloc[0]["Gene"], "Arc")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(panel["Output_Mode"]), {"direct", "twist_pcr_t7_rt"})
        self.assertEqual(len(panel), 4)

        left_direct = panel[(panel["Output_Mode"] == "direct") & (panel["Side"] == "L")].iloc[0]
        right_direct = panel[(panel["Output_Mode"] == "direct") & (panel["Side"] == "R")].iloc[0]
        left_twist = panel[(panel["Output_Mode"] == "twist_pcr_t7_rt") & (panel["Side"] == "L")].iloc[0]
        right_twist = panel[(panel["Output_Mode"] == "twist_pcr_t7_rt") & (panel["Side"] == "R")].iloc[0]

        self.assertTrue(left_direct["Final_Probe_Sequence"].endswith("CCTCAACCTACCTCCAACAATCATCTCAGTTAGT"))
        self.assertTrue(right_direct["Final_Probe_Sequence"].startswith("ACAGTTCTACGAATAATCTCACCATATTCgCTTC"))

        self.assertTrue(left_twist["Final_Probe_Sequence"].startswith("TAATACGACTCACTATAGCGTCATC"))
        self.assertTrue(right_twist["Final_Probe_Sequence"].startswith("TAATACGACTCACTATAGCGTCATCACAGTTCTACGAAT"))
        self.assertTrue(left_twist["Final_Probe_Sequence"].endswith("CGACACCGAACGTGCGACAA"))
        self.assertTrue(right_twist["Final_Probe_Sequence"].endswith("CGACACCGAACGTGCGACAA"))

    def test_optional_cross_hybridization_tm_screen_rejects_strong_offtarget(self):
        off_target_sequence = (
            "AGCGTCGATCGGCTAGCGATCGTAGCTAGCGTACGATCGTAGCTA"
            + "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
        )
        with self.fasta_path.open("w") as handle:
            handle.write(">Arc::synthetic\n")
            handle.write(self.target_sequence + "\n")
            handle.write(">OffTarget::synthetic\n")
            handle.write(off_target_sequence + "\n")

        with self.config_path.open() as handle:
            config = yaml.safe_load(handle)
        config["cycle_hcr"]["enable_cross_hybridization_tm_screen"] = True
        config["cycle_hcr"]["max_cross_hybridization_tm"] = 40
        with self.config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        result = self._run_pipeline()
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        pairs = pd.read_table(self.output_dir / "cycle_hcr_probe_pairs.tsv")
        rejected = pd.read_table(self.output_dir / "cycle_hcr_rejected_candidates.tsv")

        self.assertEqual(len(pairs), 0)
        self.assertIn("left_cross_hybridization_tm_above_max", set(rejected["Reason"]))
