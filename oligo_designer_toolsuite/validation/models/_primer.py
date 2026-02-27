from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from oligo_designer_toolsuite.validation._types import DNAT


class PrimerCycleHCR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forward_primer_sequence: Annotated[
        DNAT,
        Field(
            default="TAATACGACTCACTATAGCGTCATC",
            description="DNA sequence of the forward primer. Should be a string containing valid nucleotide characters (A, T, G, C). This primer will be placed at the 5' end of the DNA template probe during assembly. Defaults to T7 promoter sequence, change if different sequence desired.",
        ),
    ]
    reverse_primer_sequence: Annotated[
        DNAT,
        Field(
            default="CGACACCGAACGTGCGACAA",
            description="DNA sequence of the reverse primer. Should be a string containing valid nucleotide characters (A, T, G, C). This primer will be placed at the 3' end of the DNA template probe during assembly. Defaults to oligo sequence, change if different sequence desired.",
        ),
    ]
