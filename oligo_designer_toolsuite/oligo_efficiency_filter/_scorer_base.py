############################################
# imports
############################################

from abc import ABC, abstractmethod

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase

############################################
# Scorer Base Class
############################################


class BaseScorer(ABC):
    """
    Abstract base class for implementing custom oligonucleotide scoring strategies.

    Subclasses must implement the `apply` method, which computes a numeric score for a single
    oligonucleotide based on a specific criterion (e.g., GC content, melting temperature, etc.).
    """

    def __init__(self):
        """Constructor for the BaseScorer class."""

    @abstractmethod
    def apply(
        self, oligo_database: OligoDatabase, region_id: str, oligo_id: str, sequence_type: _TYPES_SEQ
    ) -> float:
        """
        Evaluate a single oligonucleotide and return a score based on a specific scoring strategy.

        This method must be implemented by all subclasses. It computes a float score for the
        oligonucleotide identified by `oligo_id` within the specified region and sequence type.

        :param oligo_database: The OligoDatabase instance containing oligonucleotide sequences and their associated properties. This database stores oligo data organized by genomic regions and can be used for filtering, property calculations, set generation, and output operations.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param oligo_id: The ID of the oligo for which the score is computed.
        :type oligo_id: str
        :param sequence_type: Type of sequence being processed. Must be one of the sequence types specified in `_constants._TYPES_SEQ`.
        :type sequence_type: _TYPES_SEQ
        :return: A float value representing the computed score for the specified oligo.
        :rtype: float
        """
