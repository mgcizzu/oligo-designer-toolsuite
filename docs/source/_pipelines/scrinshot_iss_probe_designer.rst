SCRINSHOT ISS Probe Designer
================================

This ISS variant keeps the standard SCRINSHOT workflow and changes two parts:

1. Backbone construction:
   ``sequence_padlock_backbone = gene_specific_sequence + anchor_sequence``
2. Optional 5'/3' transcript-strand flanks for each designed probe.

Detection oligos are not generated in this ISS workflow.


Command-Line Call
-----------------

::

    scrinshot_iss_probe_designer -c data/configs/scrinshot_iss_probe_designer.yaml

Cross-Hybridization Toggle
--------------------------

The ISS flow inherits SCRINSHOT target-probe specificity behavior, including the
cross-hybridization toggle:

::

    target_probe_apply_cross_hybridization: false

Behavior:

- ``true`` (default): run cross-hybridization filter.
- ``false``: skip cross-hybridization filter, while keeping exact-match and specificity/off-target filtering.


Backbone Configuration
----------------------

In ``padlock_backbone`` you provide:

- a constant ``anchor_sequence`` (used for all probes)
- ``file_gene_to_lbar`` CSV with mapping ``Gene -> Lbar_ID``
- ``file_lbar_to_sequence`` CSV with mapping ``Lbar_ID -> Sequence``

Expected CSV schemas (default column names):

Gene-to-Lbar table:

.. code-block:: text

    Gene,Lbar_ID
    AARS1,LBAR001
    DECR2,LBAR002

Lbar-to-sequence table:

.. code-block:: text

    Lbar_ID,Sequence
    LBAR001,ACGTACGT
    LBAR002,TGCATGCA

Validation performed by the ISS variant:

- required columns must exist
- ``Gene`` values must be unique in ``file_gene_to_lbar``
- ``Lbar_ID`` values must be unique in ``file_lbar_to_sequence``
- every gene-linked ``Lbar_ID`` must exist in sequence table
- sequences must contain only ``A/C/G/T``


5'/3' Flank Configuration
-------------------------

Enable and configure in ``probe_flanks``:

- ``flank_5prime_length`` = x
- ``flank_3prime_length`` = y
- ``flank_5prime_distance`` = n
- ``flank_3prime_distance`` = m

Semantics are relative to transcript strand (5'->3'):

- 5' flank starts ``n`` nt upstream of probe target start, with length ``x``
- 3' flank starts ``m`` nt downstream of probe target end, with length ``y``

Output fields added to final YAML:

- ``sequence_flank_5prime``
- ``sequence_flank_3prime``
- ``lbar_id``
- ``sequence_gene_specific``
- ``sequence_padlock_anchor``

Additional order CSV:

- ``padlock_probes_order.csv`` with columns:
  - ``Gene``
  - ``Lbar_ID``
  - ``padlock sequence``
- rows are deduplicated by default on (``Gene``, ``Lbar_ID``, ``padlock sequence``)

Additional flank CSV (same selected/deduplicated padlocks as above):

- ``padlock_probes_order_flanks.csv`` with columns:
  - ``Gene``
  - ``Lbar_ID``
  - ``padlock sequence``
  - ``flank_5prime``
  - ``flank_3prime``
