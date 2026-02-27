from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReadoutProbeCycleHCR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_readout_probe_table: Annotated[
        str,
        Field(
            default="data/functional_oligos/cycle_hcr_readout_probes_with_codebook.tsv",
            description="Path to a CSV/TSV file containing the readout probe table. The file must include columns: 'channel', 'readout_probe_id', 'L/R', and 'readout_probe_sequence'. If a 'bit' column is not present, it will be automatically assigned.",
        ),
    ]
    file_codebook: Annotated[
        str | None,
        Field(
            default="data/functional_oligos/cycle_hcr_codebook.tsv",
            description="Path to a CSV/TSV file containing an existing codebook, or None to generate a new codebook. If provided, the codebook must have region IDs as the index and bit columns (with 0/1) named 'bit_1', 'bit_2', etc. If None, a codebook will be automatically generated based on the number of regions and available readout probes.",
        ),
    ]

    @field_validator("file_readout_probe_table")
    @classmethod
    def must_be_csv_or_tsv(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.endswith(".csv") or v.endswith(".tsv")):
            raise ValueError("File must end with .csv or .tsv")
        return v
