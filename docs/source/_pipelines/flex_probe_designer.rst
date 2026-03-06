FLEX Probe Designer
===================

This pipeline designs FLEX probes for 10x GEX-style workflows on custom transcriptomes.
It reuses the Oligo-Seq target-probe design flow and then assembles FLEX constructs
with user-defined assay handles.

Command-Line Call
-----------------

::

    flex_probe_designer -c data/configs/flex_probe_designer.yaml

Input
-----

- ``files_fasta_target_probe_database``: custom transcriptome FASTA files used to generate candidate probes.
- ``files_fasta_reference_database_target_probe``: reference FASTA files for specificity filtering.
- Optional ``file_regions``: restrict design to selected gene IDs.

Cross-Hybridization Toggle
--------------------------

FLEX reuses Oligo-Seq target-probe specificity filtering and supports:

::

    target_probe_apply_cross_hybridization: false

Behavior:

- ``true`` (default): run cross-hybridization filter.
- ``false``: skip cross-hybridization filter, while keeping exact-match and specificity/off-target filtering.

FLEX Construct
--------------

FLEX target design uses two equal-length target arms. With default settings:

- ``target_arm_length: 25``
- ``target_probe_length_min/max: 50``

This yields:

- ``sequence_lhs_target_arm``: first 25 nt
- ``sequence_rhs_target_arm``: next 25 nt

Construct templates:

::

    sequence_lhs_probe = handle_5prime + linker + sequence_lhs_target_arm
    sequence_rhs_probe = sequence_rhs_target_arm + handle_3prime
    sequence_flex_probe = handle_5prime + linker + sequence_lhs_target_arm + sequence_rhs_target_arm + handle_3prime

Configure in ``flex_probe``:

- ``target_arm_length``
- ``handle_5prime``
- ``linker``
- ``handle_3prime``

GC/Tm Filtering Behavior
------------------------

FLEX disables whole-target (50 nt) GC/Tm filtering and neutralizes GC/Tm scoring during
target search and set selection. This avoids biasing by full-length probe properties.

Instead, FLEX applies GC constraints per target arm:

- ``flex_probe.target_arm_gc_filter.enabled`` (default ``true``)
- ``flex_probe.target_arm_gc_filter.gc_content_min`` (default ``44``)
- ``flex_probe.target_arm_gc_filter.gc_content_max`` (default ``72``)

Each probe is retained only if both ``sequence_lhs_target_arm`` and
``sequence_rhs_target_arm`` satisfy the configured GC range.

Ligation Junction Constraints
-----------------------------

By default, the pipeline applies a ligation-junction filter aligned with the
10x recommendation:

- enforce ``T`` at probe position 25 (1-based)
- allow any base at position 26 (TN motif)

Configuration:

- ``flex_probe.ligation_filter.enabled``
- ``flex_probe.ligation_filter.lhs_position`` (default ``25``)
- ``flex_probe.ligation_filter.required_lhs_base`` (default ``T``)
- ``flex_probe.ligation_filter.required_lhs_mode`` (``hard``|``soft``|``off``; default ``hard``)
- ``flex_probe.ligation_filter.prohibited_ligation_pairs`` (list of 2-mer probe pairs to exclude)

Output
------

In addition to standard target-probe attributes, output includes:

- ``sequence_target_probe``
- ``sequence_lhs_target_arm``
- ``sequence_rhs_target_arm``
- ``sequence_lhs_probe``
- ``sequence_rhs_probe``
- ``sequence_flex_probe``
- ``sequence_handle_5prime``
- ``sequence_linker``
- ``sequence_handle_3prime``
- ``target_arm_length``
- ``gc_content_lhs_target_arm``
- ``gc_content_rhs_target_arm``
- ``gc_content_target_arm_min``
- ``gc_content_target_arm_max``
- ``ligation_lhs_position``
- ``ligation_lhs_base``
- ``ligation_rhs_base``
- ``ligation_probe_pair``
- ``ligation_required_lhs_mode``
- ``ligation_required_lhs_base``
- ``ligation_required_lhs_pass``
