# Standard Parameter Descriptions

This document defines the standard descriptions for common parameters used throughout the oligo-designer-toolsuite package. These descriptions should be used consistently across all docstrings.

## Common Parameters

### `oligo_database`
**Standard description:**
```
The OligoDatabase instance containing oligonucleotide sequences and their associated properties.
This database stores oligo data organized by genomic regions and can be used for filtering,
property calculations, set generation, and output operations.
```

### `dir_output`
**Standard description:**
```
Directory path where output files will be saved.
```

### `n_jobs`
**Standard description:**
```
Number of parallel jobs to use for processing.
```

### `reference_database`
**Standard description:**
```
The ReferenceDatabase instance containing reference sequences for alignment or comparison.
```

### `region_ids`
**Standard description:**
```
Region identifier(s) to process. Can be a single region ID (str) or a list of region IDs (List[str]).
If None, all regions in the database are processed.
```

### `files_fasta`
**Standard description:**
```
Path(s) to FASTA file(s) to load. Can be a single file path (str) or a list of file paths (List[str]).
```

### `sequence_type`
**Standard description:**
```
Type of sequence being processed. Must be one of the sequence types specified in `_constants._TYPES_SEQ`.
```
