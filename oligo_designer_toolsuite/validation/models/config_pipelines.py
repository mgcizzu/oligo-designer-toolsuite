from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from oligo_designer_toolsuite.validation.models._developer_parameters import DeveloperParametersCycleHCR
from oligo_designer_toolsuite.validation.models._general import General
from oligo_designer_toolsuite.validation.models._primer import PrimerCycleHCR
from oligo_designer_toolsuite.validation.models._readout_probes import ReadoutProbeCycleHCR
from oligo_designer_toolsuite.validation.models._target_probes import TargetProbeCycleHCR


class CycleHCRProbeDesignerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: PositiveInt
    general: Annotated[
        General,
        Field(
            default_factory=lambda: General(
                n_jobs=2, dir_output="output_cyclehcr_probe_designer", top_n_sets=3
            ),
            description="General parameters of the pipeline.",
        ),
    ]
    target_probe: TargetProbeCycleHCR
    readout_probe: ReadoutProbeCycleHCR
    primer: PrimerCycleHCR
    developer_param: DeveloperParametersCycleHCR
