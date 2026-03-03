from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PositiveInt

from oligo_designer_toolsuite.validation.models._developer_parameters import (
    DeveloperParametersCycleHCR,
    DeveloperParametersMerfish,
)
from oligo_designer_toolsuite.validation.models._general import General
from oligo_designer_toolsuite.validation.models._primer import PrimerCycleHCR, PrimerMerfish
from oligo_designer_toolsuite.validation.models._readout_probes import (
    ReadoutProbeCycleHCR,
    ReadoutProbeMerfish,
)
from oligo_designer_toolsuite.validation.models._target_probes import TargetProbeCycleHCR, TargetProbeMerfish


class CycleHCRProbeDesignerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: PositiveInt
    general: General
    target_probe: TargetProbeCycleHCR
    readout_probe: ReadoutProbeCycleHCR
    primer: PrimerCycleHCR
    developer_param: DeveloperParametersCycleHCR


class MerfishProbeDesignerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: PositiveInt
    general: General
    target_probe: TargetProbeMerfish
    readout_probe: ReadoutProbeMerfish
    primer: PrimerMerfish
    developer_param: DeveloperParametersMerfish
