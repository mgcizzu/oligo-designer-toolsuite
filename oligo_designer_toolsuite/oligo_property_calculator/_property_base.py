############################################
# imports
############################################

from abc import ABC, abstractmethod

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase

############################################
# Property Base Class
############################################


class BaseProperty(ABC):
    """
    An abstract base class for creating oligo property calculators.

    The `BaseProperty` class serves as a template for developing property calculators that compute
    specific attributes of oligonucleotides. Subclasses must implement the `apply` method to define
    the property calculation logic.
    """

    def __init__(self) -> None:
        """Constructor for the BaseProperty class."""

    @abstractmethod
    def apply(
        self, oligo_database: OligoDatabase, region_id: str, oligo_id: str, sequence_type: _TYPES_SEQ
    ) -> dict:
        """
        Calculate the property for a specific oligo and return the result as a dictionary.

        This abstract method must be implemented by subclasses to define the specific property
        calculation logic for a given oligo.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param oligo_id: The ID of the oligo for which the property is calculated.
        :type oligo_id: str
        :param sequence_type: The type of sequence to be used for property calculation.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        :return: A dictionary containing the calculated property(ies). Keys are attribute names, values are the calculated values.
        :rtype: dict
        """
