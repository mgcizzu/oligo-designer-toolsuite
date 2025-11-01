############################################
# imports
############################################

from typing import List, get_args

from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_property_calculator import BaseProperty

############################################
# Property Calculator Class
############################################


class PropertyCalculator:
    """
    A class for applying multiple property calculators to oligonucleotides in an OligoDatabase.

    The `PropertyCalculator` class allows you to apply a list of property calculators (subclasses of `BaseProperty`) to an OligoDatabase.
    The properties are calculated in parallel across all regions of the database, and the calculated values are stored as attributes.

    :param properties: A list of property calculators to apply to oligonucleotides.
    :type properties: List[BaseProperty]
    """

    def __init__(self, properties: List[BaseProperty]) -> None:
        """Constructor for the PropertyCalculator class."""
        self.properties = properties

    def apply(
        self, oligo_database: OligoDatabase, sequence_type: _TYPES_SEQ, n_jobs: int = 1
    ) -> OligoDatabase:
        """
        Apply the property calculators to all oligonucleotides in the OligoDatabase and update
        the database with the calculated property values.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param sequence_type: The type of sequence to be used for property calculation.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        :param n_jobs: The number of jobs to run in parallel, defaults to 1.
        :type n_jobs: int
        :return: The updated OligoDatabase with the calculated properties.
        :rtype: OligoDatabase
        """
        options = get_args(_TYPES_SEQ)
        assert (
            sequence_type in options
        ), f"Sequence type not supported! '{sequence_type}' is not in {options}."

        region_ids = list(oligo_database.database.keys())
        with joblib_progress(description="Property Calculator", total=len(region_ids)):
            Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
                delayed(self._calculate_region)(oligo_database, region_id, sequence_type)
                for region_id in region_ids
            )

        return oligo_database

    def _calculate_region(
        self, oligo_database: OligoDatabase, region_id: str, sequence_type: _TYPES_SEQ
    ) -> None:
        """
        Calculate properties for all oligonucleotides in a specific region of the OligoDatabase.

        This method iterates through the oligonucleotides in a given region of the database,
        applying all property calculators to each oligo and updating the database with the calculated values.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param sequence_type: The type of sequence to be used for the property calculations.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        """
        new_oligo_attribute = {}

        for oligo_id in oligo_database.database[region_id].keys():
            # Calculate all properties for this oligo
            for property_calc in self.properties:
                property_result = property_calc.apply(
                    oligo_database=oligo_database,
                    region_id=region_id,
                    oligo_id=oligo_id,
                    sequence_type=sequence_type,
                )
                # Merge results into the attribute dictionary
                if oligo_id not in new_oligo_attribute:
                    new_oligo_attribute[oligo_id] = {}
                new_oligo_attribute[oligo_id].update(property_result)

        # Update all oligo attributes at once for this region
        oligo_database.update_oligo_attributes(new_oligo_attribute)
