############################################
# imports
############################################

from typing import Literal

############################################
# types
############################################

_TYPES_SEQ = Literal[
    "target",
    "target_short",
    "oligo",
    "oligo_short",
    "oligo_pair_L",
    "oligo_pair_R",
    "sequence_encoding_probe",
    "sequence_target_probe",
    # Reverse complement variants
    "target_rc",
    "target_short_rc",
    "oligo_rc",
    "oligo_short_rc",
    "oligo_pair_L_rc",
    "oligo_pair_R_rc",
    "sequence_encoding_probe_rc",
    "sequence_target_probe_rc",
]
_TYPES_REF = Literal["fasta", "vcf"]
_TYPES_FILE = Literal["gff", "gtf", "fasta"]
_TYPES_FILE_SEQ = Literal["dna", "ncrna"]

############################################
# constants
############################################

SEPARATOR_OLIGO_ID = "::"
SEPARATOR_FASTA_HEADER_FIELDS = "::"
SEPARATOR_FASTA_HEADER_FIELDS_LIST = ";"
