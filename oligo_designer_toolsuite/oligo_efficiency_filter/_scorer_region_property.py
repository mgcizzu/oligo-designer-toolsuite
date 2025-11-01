############################################
# imports
############################################

from oligo_designer_toolsuite._constants import _TYPES_SEQ
from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_efficiency_filter import BaseScorer
from oligo_designer_toolsuite.oligo_property_calculator._property_functions import calc_isoform_consensus
from oligo_designer_toolsuite.utils import check_if_list

############################################
# Sequence Property Scorer Classes
############################################


class OverlapTargetedExonsScorer(BaseScorer):
    """
    Scores oligos based on whether they overlap with a set of targeted exons.

    If an oligo overlaps at least one of the specified targeted exons, it receives a weighted score.
    This can be used to prioritize probes targeting specific exonic regions.

    :param targeted_exons: List of exon identifiers that should be targeted by oligos.
    :type targeted_exons: list
    :param score_weight: Weight to apply if an oligo overlaps with a targeted exon.
    :type score_weight: float
    """

    def __init__(self, targeted_exons: list, score_weight: float):
        """Constructor for the OverlapTargetedExonsScorer class."""

        self.targeted_exons = sorted(targeted_exons)
        self.score_weight = score_weight

    def apply(self, oligo_database: OligoDatabase, region_id: str, oligo_id: str, sequence_type: _TYPES_SEQ):
        """
        Apply the targeted exon overlap scoring strategy to a given oligo.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param oligo_id: The ID of the oligo for which the score is computed.
        :type oligo_id: str
        :param sequence_type: The type of sequence to be used for filter calculations. Not used in this function.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        :return: Weighted score based on overlap with targeted exons.
        :rtype: float
        """
        exon_numbers = oligo_database.get_oligo_attribute_value(
            "exon_number", flatten=True, region_id=region_id, oligo_id=oligo_id
        )
        exon_numbers = check_if_list(exon_numbers)

        if exon_numbers is None:
            in_targeted_exons = False
        else:
            in_targeted_exons = any(
                any(exon in self.targeted_exons for exon in str(exon_number).split("__JUNC__"))
                for exon_number in exon_numbers
            )

        score = self.score_weight * in_targeted_exons

        return score


class OverlapUTRScorer(BaseScorer):
    """
    Scores oligos based on whether they originate from untranslated regions (UTRs).

    Oligos that map to either 5' or 3' UTRs receive a weighted score. This is useful when
    targeting regulatory regions of transcripts.

    :param score_weight: Weight to apply if an oligo originates from a UTR.
    :type score_weight: float
    """

    def __init__(self, score_weight: float):
        """Constructor for the OverlapUTRScorer class."""

        self.score_weight = score_weight

    def apply(self, oligo_database: OligoDatabase, region_id: str, oligo_id: str, sequence_type: _TYPES_SEQ):
        """
        Apply the UTR overlap scoring strategy to a given oligo.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param oligo_id: The ID of the oligo for which the score is computed.
        :type oligo_id: str
        :param sequence_type: The type of sequence to be used for filter calculations. Not used in this function.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        :return: Weighted score based on UTR overlap.
        :rtype: float
        """
        regiontype = oligo_database.get_oligo_attribute_value(
            attribute="regiontype", region_id=region_id, oligo_id=oligo_id, flatten=True
        )
        if regiontype:
            sequence_originates_from_UTR = "three_prime_UTR" in regiontype or "five_prime_UTR" in regiontype
        else:
            sequence_originates_from_UTR = False

        score = self.score_weight * sequence_originates_from_UTR

        return score


class IsoformConsensusScorer(BaseScorer):
    """
    Scores oligos based on their presence across transcript isoforms (isoform consensus).

    Oligos that are found in more transcript isoforms receive a lower score (after normalization),
    allowing the user to favor isoform-specific or consensus-targeting designs.

    :param normalize: Whether to normalize the consensus score to a range of 0–1.
    :type normalize: bool
    :param score_weight: Weight to apply to the consensus score.
    :type score_weight: float
    """

    def __init__(self, normalize: bool, score_weight: float):
        """Constructor for the IsoformConsensusScorer class."""

        self.normalize = normalize
        self.score_weight = score_weight

    def apply(self, oligo_database: OligoDatabase, region_id: str, oligo_id: str, sequence_type: _TYPES_SEQ):
        """
        Apply the isoform consensus scoring strategy to a given oligo.

        :param oligo_database: The OligoDatabase containing the oligonucleotides and their associated attributes.
        :type oligo_database: OligoDatabase
        :param region_id: Region ID to process.
        :type region_id: str
        :param oligo_id: The ID of the oligo for which the score is computed.
        :type oligo_id: str
        :param sequence_type: The type of sequence to be used for filter calculations. Not used in this function.
        :type sequence_type: _TYPES_SEQ["oligo", "target"]
        :return: Weighted score based on isoform consensus.
        :rtype: float
        """
        transcript_id = oligo_database.get_oligo_attribute_value(
            attribute="transcript_id", region_id=region_id, oligo_id=oligo_id, flatten=True
        )
        number_transcripts = oligo_database.get_oligo_attribute_value(
            attribute="number_total_transcripts", region_id=region_id, oligo_id=oligo_id, flatten=True
        )
        if transcript_id and number_transcripts:
            isoform_consensus = calc_isoform_consensus(
                transcript_id=transcript_id, number_total_transcripts=number_transcripts
            )
            if self.normalize:
                # isoform consensus is given in % (0-100), hence we devide by 100
                # we use 1 - isoform consensus as normalized score
                isoform_consensus = 1 - (isoform_consensus / 100)
        else:
            # if information not available, don't consider isoform consensus in scoring
            isoform_consensus = 0
        score = self.score_weight * isoform_consensus

        return score
