CycleHCR Probe Designer
=======================

CycleHCR probe design combines transcript-targeting probe-pair selection with explicit barcode assignment and optional Twist-compatible pool assembly.
Each accepted target window spans 92 nt and is split into a 45-nt left arm, a 2-nt untargeted gap, and a 45-nt right arm.

The current repository file set documents the agreed input contract and output assemblies for a future ``CycleHCR`` pipeline implementation.
It does not yet imply that the pipeline class or command-line entrypoint exists.

Design Summary
--------------

Target selection follows a sliding-window search over transcript sequences.
For each candidate 92-nt window:

- the left and right 45-nt target segments are evaluated independently
- each 45-mer must have DNA:DNA melting temperature <= 76 C
- each 45-mer must have GC content between 30% and 90%
- each 45-mer must avoid homopolymeric runs of length 6
- each 45-mer must satisfy an estimated RNA:DNA binding Tm >= 90 C, using ``Tm(probe:RNA) = Tm(probe:DNA) + 10 C``
- optionally, candidates can pass a transcriptome-wide cross-hybridization Tm screen with a CycleHCR threshold of <= 72 C
- final probe pairs are filtered by a transcriptome-wide uniqueness check on a 26-nt junction sequence

Accepted probe pairs are then assembled with gene-level barcodes loaded from an HCR barcode workbook.

Barcode Assignment
------------------

The barcode library is provided as an Excel workbook, for example ``HCR_barcodes.xlsx``.
Each sheet represents one barcode family, and each row provides a left barcode name and sequence plus a right barcode name and sequence.

Barcode assignment is explicit:

- the user provides a CSV mapping ``Gene, Sheet, Left_Barcode_Name, Right_Barcode_Name``
- each gene receives exactly one barcode combination
- the same ``(Sheet, Left_Barcode_Name, Right_Barcode_Name)`` combination must not be reused across genes in the same panel
- left and right barcode names for one gene must come from the same sheet

An example mapping file is provided at ``data/barcodes/cycle_hcr_gene_barcode_assignment.csv``.

Output Modes
------------

Two output assemblies are currently defined.

Direct targeting oligos
~~~~~~~~~~~~~~~~~~~~~~~

When probes are ordered directly for imaging, the target-binding domains must already be antisense to the target transcript.

- left primary: ``RC(target_left_45) + TT + full_left_barcode``
- right primary: ``full_right_barcode + TT + RC(target_right_45)``

Twist PCR/T7/RT pool constructs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When probes are ordered as pool constructs and converted through ``PCR -> T7 transcription -> reverse transcription``, the ordered DNA contains the sense target segments so that the final RT-derived cDNA becomes antisense to the target transcript.

The default construct blocks are:

- forward PCR/T7 block: ``TAATACGACTCACTATAGCGTCATC``
- reverse block: ``CGACACCGAACGTGCGACAA``
- spacer: ``TT``

Barcode extraction rules:

- left barcode subsequence: terminal 14 nt of the selected left barcode sequence
- right barcode subsequence: leading 14 nt of the selected right barcode sequence

Assembled orderable constructs:

- left Twist construct: ``forward_PCR_T7 + target_left_45 + TT + left_barcode_14 + reverse_block``
- right Twist construct: ``forward_PCR_T7 + right_barcode_14 + TT + target_right_45 + reverse_block``

Configuration Files
-------------------

The design contract is captured in the following files:

- ``data/configs/cycle_hcr_probe_designer.yaml``: full configuration schema for the proposed pipeline
- ``data/barcodes/cycle_hcr_gene_barcode_assignment.csv``: example explicit barcode assignment file

Suggested outputs for the future implementation include:

- ``cycle_hcr_probe_panel.tsv``: one row per final oligo sequence
- ``cycle_hcr_probe_pairs.tsv``: one row per probe pair
- ``cycle_hcr_barcode_assignments.tsv``: one row per gene barcode assignment
- ``cycle_hcr_rejected_candidates.tsv``: failed windows with rejection reasons

Optional Cross-Hybridization Tm Screen
--------------------------------------

The configuration now supports an optional transcriptome-wide cross-hybridization Tm screen:

- ``cycle_hcr.enable_cross_hybridization_tm_screen``: enables or disables the screen
- ``cycle_hcr.cross_hybridization_seed_length``: exact seed length used to find transcriptome windows that are plausible off-targets

When enabled, the pipeline builds a seed index over the reference transcriptome, gathers candidate off-target windows that share an exact seed with each 45-nt arm, estimates duplex Tm for those windows, and rejects the arm if the worst off-target Tm exceeds ``cycle_hcr.max_cross_hybridization_tm``.
Exact off-target matches use the configured DNA nearest-neighbor Tm model directly; seeded mismatch candidates currently use a similarity-scaled heuristic estimate rather than a full mismatch thermodynamic model.
