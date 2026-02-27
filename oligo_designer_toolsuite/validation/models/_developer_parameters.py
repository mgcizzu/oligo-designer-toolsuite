from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from oligo_designer_toolsuite.validation._types import SecondaryStructuresThresholdDeltaGT
from oligo_designer_toolsuite.validation.models._general import (
    BlastnHitParameters,
    BlastnSearchParameters,
    OligoSetSelection,
    TmChemCorrectionParameters,
    TmParameters,
    TmSaltCorrectionParameters,
)


class TargetProbeDev(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secondary_structures_threshold_deltaG: SecondaryStructuresThresholdDeltaGT
    specificity_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            description="BLASTN search parameters for specificity filtering. These parameters control how BLASTN searches are performed to identify off-target binding sites."
        ),
    ]
    specificity_blastn_hit_parameters: Annotated[
        BlastnHitParameters,
        Field(
            description="Parameters for filtering BLASTN hits during specificity analysis. Use either coverage or min_alignment_length."
        ),
    ]
    cross_hybridization_blastn_search_parameters: Annotated[BlastnSearchParameters, Field(description="")]
    cross_hybridization_blastn_hit_parameters: BlastnHitParameters


class TargetProbeDevCycleHCR(TargetProbeDev):

    Tm_parameters: Annotated[
        TmParameters,
        Field(
            description="Parameters for calculating melting temperature (Tm) using the nearest-neighbor method. For more information on parameters, see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.Tm_NN"
        ),
    ]
    Tm_chem_correction_parameters: Annotated[
        TmChemCorrectionParameters,
        Field(
            description="Optional parameters for chemical correction.  For more information on parameters, see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.chem_correction"
        ),
    ]
    Tm_salt_correction_parameters: Annotated[
        TmSaltCorrectionParameters,
        Field(
            description="Optional parameters to account for the effects of salt concentration on melting temperature. For more information on parameters, see: https://biopython.org/docs/1.75/api/Bio.SeqUtils.MeltingTemp.html#Bio.SeqUtils.MeltingTemp.salt_correction"
        ),
    ]

    cross_hybridization_blastn_search_parameters: Annotated[
        BlastnSearchParameters,
        Field(
            description="Pydantic model of BLASTN search parameters for cross-hybridization filtering. These parameters control how BLASTN searches are performed to identify potential cross-hybridization between left and right probe pairs within the same set."
        ),
    ]
    cross_hybridization_blastn_hit_parameters: Annotated[
        BlastnHitParameters,
        Field(
            description="Parameters for filtering BLASTN hits in cross-hybridization searches. Use either coverage or min_alignment_length. Probes with cross-hybridization hits meeting these criteria are removed from the larger region."
        ),
    ]


class DeveloperParametersBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oligo_set_selection: OligoSetSelection


class DeveloperParametersCycleHCR(DeveloperParametersBase):
    target_probe: TargetProbeDevCycleHCR
