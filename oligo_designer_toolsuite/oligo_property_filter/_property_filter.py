############################################
# imports
############################################

from typing import get_args

from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_property_filter import BasePropertyFilter

############################################
# Property Filter Class
############################################


class PropertyFilter:
    """
    A class for applying multiple property filters to sequences in an OligoDatabase.

    The `PropertyFilter` class allows you to apply a list of sequence filters (subclasses of `BasePropertyFilter`) to an OligoDatabase.
    The filters are applied in parallel across all regions of the database, and sequences that do not meet all filter criteria are removed.

    :param filters: A list of property filters to apply to sequences.
    :type filters: list[BasePropertyFilter]
    """

    def __init__(
        self,
        filters: list[BasePropertyFilter],
    ) -> None:
        """Constructor for the PropertyFilter class."""
        self.filters = filters

    def apply(
        self, oligo_database: OligoDatabase, sequence_type: _TYPES_SEQ, n_jobs: int = 1
    ) -> OligoDatabase:
        """
        Apply the property filters to all sequences in the OligoDatabase and filter
        sequences in the OligoDatabase based on the specified property filters.
        Sequences that do not meet the criteria of all filters are removed.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param sequence_type: Type of sequence being processed. Must be one of the sequence types specified in `_constants._TYPES_SEQ`.
        :type sequence_type: _TYPES_SEQ
        :param n_jobs: Number of parallel jobs to use for processing. Defaults to 1.
        :type n_jobs: int
        :return: The filtered OligoDatabase.
        :rtype: OligoDatabase
        """
        options = get_args(_TYPES_SEQ)
        assert (
            sequence_type in options
        ), f"Sequence type not supported! '{sequence_type}' is not in {options}."

        region_ids = list(oligo_database.database.keys())
        with joblib_progress(description="Property Filter", total=len(region_ids)):
            Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
                delayed(self._filter_region)(oligo_database, region_id, sequence_type)
                for region_id in region_ids
            )

        return oligo_database

    def _filter_region(
        self, oligo_database: OligoDatabase, region_id: str, sequence_type: _TYPES_SEQ
    ) -> None:
        """
        Filters a specific region in the OligoDatabase based on sequence properties.

        This method iterates through the oligonucleotides in a given region of the database,
        applying a series of filters to determine whether each sequence meets specified criteria.
        If a sequence does not fulfill all filter conditions, it is removed from the database.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param sequence_type: Type of sequence being processed. Must be one of the sequence types specified in `_constants._TYPES_SEQ`.
        :type sequence_type: _TYPES_SEQ
        """
        oligo_ids = list(oligo_database.database[region_id].keys())
        for oligo_id in oligo_ids:
            fulfills_all_filter = self._filter_sequence(
                oligo_database.database[region_id][oligo_id][sequence_type]
            )
            if not fulfills_all_filter:
                del oligo_database.database[region_id][oligo_id]

    def _filter_sequence(self, sequence: str) -> bool:
        """
        Applies a series of filters to a sequence and checks if it meets all criteria.

        This method iterates through a list of filters, applying each one to the provided sequence.
        If the sequence fails any filter, the method returns `False` immediately,
        indicating that the sequence does not meet the criteria.
        If the sequence passes all filters, it returns `True`.

        :param sequence: The nucleotide sequence.
        :type sequence: str
        :return: `True` if the sequence meets all filter criteria, `False` otherwise.
        :rtype: bool
        """
        for filter in self.filters:
            fulfills_filter = filter.apply(sequence)
            if not fulfills_filter:  # stop at the first false we obtain
                return False
        return True
