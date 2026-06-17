Custom-Sequence SCRINSHOT ISS Probe Designer
============================================

This pipeline is intended for targets that are not part of the reference
transcriptome, such as transgenes, GFP variants, reporters, or synthetic
constructs.

The workflow is:

1. Load one custom FASTA sequence in transcript/sense orientation.
2. Generate candidate target sites with the SCRINSHOT sliding-window rules.
3. Apply the standard sequence-property and padlock-arm filters.
4. BLAST candidates against a reference transcriptome.
5. Keep only candidates with no configured BLAST hit.
6. Select probe sets and build ISS padlocks.


Command-Line Call
-----------------

::

    custom_sequence_scrinshot_iss_probe_designer -c data/configs/custom_sequence_scrinshot_iss_probe_designer.yaml


Target FASTA
------------

``files_fasta_target_probe_database`` should normally contain one FASTA record.
The header can be plain:

.. code-block:: text

    >custom_reporter
    ATGGCT...

Coordinate-free FASTA records are assigned synthetic 1-based coordinates for
candidate spacing, output traceability, and optional flank extraction. These are
not genomic coordinates.


Transcriptome Zero-Hit Filter
-----------------------------

``files_fasta_reference_database_target_probe`` points to the transcriptome or
other reference sequences that should be avoided. A candidate is removed when
BLAST reports a hit passing ``target_probe_specificity_blastn_hit_parameters``.

The default config keeps the existing SCRINSHOT convention:

.. code-block:: yaml

    target_probe_specificity_blastn_search_parameters:
      strand: "minus"

Use this when the custom FASTA and reference transcriptome are both in
transcript/sense orientation, because the query oligo is the reverse complement
of the target site.


Backbone Configuration
----------------------

For a single reporter, the simplest configuration is a direct gene-specific
sequence:

.. code-block:: yaml

    padlock_backbone:
      anchor_sequence: TGCGTCTATTTAGTGGAGCC
      gene_specific_sequence: ACGTACGTACGTACGTACGT
      direct_lbar_id: custom_reporter

The inherited ISS CSV mapping remains available through ``file_gene_to_lbar``
and ``file_lbar_to_sequence`` if multiple targets or existing Lbar assignments
are preferred.


Outputs
-------

The output files match the SCRINSHOT ISS format:

- ``padlock_probes.yml``
- ``padlock_probes_order.yml``
- ``padlock_probes_order.csv``
- ``padlock_probes_order_flanks.csv`` when flank columns are requested
