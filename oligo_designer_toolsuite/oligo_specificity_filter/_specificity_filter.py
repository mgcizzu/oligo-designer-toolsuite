############################################
# imports
############################################

from typing import get_args

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_specificity_filter import BaseSpecificityFilter

############################################
# Specificity Filter Classe
############################################


class SpecificityFilter:
    """
    A class for managing and applying a sequence of specificity filters to an OligoDatabase.

    The `SpecificityFilter` class aggregates multiple filters, each of which applies specific criteria to determine
    the suitability of oligonucleotides based on their sequence specificity. This allows for the sequential application
    of various filtering methods to refine the OligoDatabase according to specificity requirements.

    :param filters: A list of filters, each inheriting from the `BaseSpecificityFilter`, that will be applied to the OligoDatabase.
    :type filters: list[BaseSpecificityFilter]
    """

    def __init__(
        self,
        filters: list[BaseSpecificityFilter],
    ) -> None:
        """Constructor for the SpecificityFilter class."""
        self.filters = filters

    def apply(
        self,
        oligo_database: OligoDatabase,
        sequence_type: _TYPES_SEQ = None,
        n_jobs: int = 1,
    ) -> OligoDatabase:
        """
        Applies a sequence of specificity filters to an OligoDatabase.

        The `apply` method processes the provided OligoDatabase through each specificity filter in sequence.
        It evaluates the database against reference sequences, if provided, and ensures that only oligonucleotides
        meeting all specificity criteria are retained.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param sequence_type: Type of sequence being processed. Must be one of the sequence types specified in `_constants._TYPES_SEQ`.
        :type sequence_type: _TYPES_SEQ
        :param n_jobs: Number of parallel jobs to use for processing.
        :type n_jobs: int
        :return: The filtered OligoDatabase.
        :rtype: OligoDatabase
        """
        options = get_args(_TYPES_SEQ)
        assert (
            sequence_type in options
        ), f"Sequence type not supported! '{sequence_type}' is not in {options}."

        for specificity_filter in self.filters:
            oligo_database = specificity_filter.apply(
                oligo_database=oligo_database,
                sequence_type=sequence_type,
                n_jobs=n_jobs,
            )

        return oligo_database
