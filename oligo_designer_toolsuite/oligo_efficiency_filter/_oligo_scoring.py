############################################
# imports
############################################

from typing import List, Tuple, get_args

import pandas as pd

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase

from ._scorer_base import BaseScorer

############################################
# Oligo Scoring Classes
############################################


class OligoScoring:
    """
    Applies a set of scoring strategies to oligonucleotides in a given region.

    This class takes a list of scorer objects (implementing the BaseScorer interface) and applies
    them to all oligos within a specified region and sequence type. Each scorer contributes to the
    total score of an oligo, which is saved in the database and returned as a pandas Series.

    :param scorers: A list of scorer instances that define how each oligo should be evaluated.
    :type scorers: List[BaseScorer]
    """

    def __init__(self, scorers: List[BaseScorer]):
        """Constructor for the OligoScoring class."""
        self.scorers = scorers

    def apply(
        self, oligo_database: OligoDatabase, region_id: str, sequence_type: _TYPES_SEQ
    ) -> Tuple[OligoDatabase, pd.Series]:
        """
        Apply all configured scorers to the oligonucleotides within a given region and sequence type.

        The method iterates through all oligos in the specified region and computes a cumulative score
        for each oligo using the list of scoring strategies provided at initialization. The scores are
        added to the oligo entries under the key 'oligo_score' and also returned as a pandas Series.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param sequence_type: The type of sequence to be used for filter calculations.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        :return: A tuple containing the updated OligoDatabase and a pandas Series of scores indexed by oligo ID.
        :rtype: Tuple[OligoDatabase, pd.Series]
        """

        options = get_args(_TYPES_SEQ)
        assert (
            sequence_type in options
        ), f"Sequence type not supported! '{sequence_type}' is not in {options}."

        oligos_ids = list(oligo_database.database[region_id].keys())
        oligos_scores = pd.Series(index=oligos_ids, dtype=float)
        for oligo_id in oligos_ids:
            score = 0.0
            for scorer in self.scorers:
                score += scorer.apply(
                    oligo_database=oligo_database,
                    region_id=region_id,
                    oligo_id=oligo_id,
                    sequence_type=sequence_type,
                )
            oligo_database.database[region_id][oligo_id]["oligo_score"] = round(score, 4)
            oligos_scores[oligo_id] = round(score, 4)
        return oligo_database, oligos_scores
