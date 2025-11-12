<!-- 2149ad8a-b051-4d54-8c89-bc95cea41b4f 3eb45cf9-e766-48ca-9d17-e8bd19246218 -->
# Refactor OligoAttributes to Property Calculator System - Step-by-Step Plan

## Overview

Refactor `oligo_designer_toolsuite/database/_oligo_database_attributes.py` to follow the same architectural pattern as `oligo_property_filter` and `oligo_efficiency_filter` modules. The refactoring will be done incrementally, with each step verified before proceeding.

## Module Structure

### New Module: `oligo_property_calculator/`

```
oligo_property_calculator/
├── __init__.py
├── _property_base.py          # BaseProperty abstract base class
├── _property_calculator.py   # PropertyCalculator class (main orchestrator)
├── _property_functions.py     # Shared calculation functions (used by filters and calculators)
├── _property_sequence.py      # Sequence-based properties (GC, Tm, length, etc.)
└── _property_region.py       # Region-based properties (isoform consensus, num transcripts, etc.)
```

## Step-by-Step Implementation Plan

### PHASE 1: Setup and Foundation (Steps 1-5)

These steps create the basic structure without breaking existing code.

**Step 1: Create Module Directory and Base Files**

- Create `oligo_property_calculator/` directory
- Create empty `__init__.py` file
- Create `_property_base.py` with placeholder BaseProperty class
- Verify: New module can be imported, no syntax errors

**Step 2: Extract and Refactor Calculation Functions - Part 1 (Core Functions)**

- Create `_property_functions.py`
- Extract first 5-7 core calculation functions:
  - `calc_oligo_length` (from `_calc_oligo_length`)
  - `calc_gc_content` (from `_calc_GC_content`, rename)
  - `calc_tm_nn` (from `_calc_TmNN`, rename)
  - `calc_dg_secondary_structure` (from `_calc_DG_secondary_structure`, rename)
  - `calc_length_complement` (from `_calc_length_complement`)
  - `calc_length_selfcomplement` (new, using `calc_length_complement`)
- Keep original functions in `OligoAttributes` for now (backward compatibility)
- Verify: New functions produce same results as old static methods

**Step 3: Extract and Refactor Calculation Functions - Part 2 (Remaining Sequence Functions)**

- Add remaining sequence-related functions:
  - `calculate_shortened_sequence` (from `_calculate_shortened_sequence`)
  - `calculate_reverse_complement_sequence` (from `_calculate_reverse_complement_sequence`)
  - `calc_split_sequence` (from `_calc_split_sequence`)
  - `calc_seedregion` (from `_calc_seedregion`)
  - `calculate_seedregion_site` (from `_calculate_seedregion_site`)
- Verify: All sequence functions extracted and tested

**Step 4: Extract and Refactor Calculation Functions - Part 3 (Complex Functions)**

- Add complex calculation functions:
  - `calc_padlock_arms` (from `_calc_padlock_arms`)
  - `calc_detect_oligo` (from `_calc_detect_oligo`)
- Verify: Complex functions work correctly

**Step 5: Extract and Refactor Calculation Functions - Part 4 (Region Functions)**

- Add region-related functions:
  - `calc_num_targeted_transcripts` (from `_calc_num_targeted_transcripts`)
  - `calc_isoform_consensus` (from `_calc_isoform_consensus`)
- Verify: All functions extracted, `_property_functions.py` complete

### PHASE 2: Create Property Classes (Steps 6-10)

These steps create the property classes that use the extracted functions.

**Step 6: Implement BaseProperty Class**

- Complete `_property_base.py` with full `BaseProperty` abstract class
- Abstract method signature: `apply(oligo_database, region_id, oligo_id, sequence_type) -> dict`
- Verify: BaseProperty can be imported and subclassed

**Step 7: Implement Sequence Property Classes - Part 1 (Simple Properties)**

- Create `_property_sequence.py`
- Implement first 5 simple property classes:
  - `LengthProperty`
  - `GCContentProperty`
  - `TmNNProperty`
  - `DGSecondaryStructureProperty`
  - `LengthSelfComplementProperty`
- Each class uses corresponding function from `_property_functions.py`
- Verify: Each property class can calculate and return correct values

**Step 8: Implement Sequence Property Classes - Part 2 (Complex Properties)**

- Add remaining sequence property classes:
  - `LengthComplementProperty`
  - `ShortenedSequenceProperty`
  - `ReverseComplementSequenceProperty`
  - `SplitSequenceProperty`
  - `SeedregionProperty`
  - `SeedregionSiteProperty`
- Verify: All sequence properties implemented

**Step 9: Implement Sequence Property Classes - Part 3 (Specialized Properties)**

- Add specialized property classes:
  - `PadlockArmsProperty`
  - `DetectOligoProperty`
- Verify: Specialized properties work correctly

**Step 10: Implement Region Property Classes**

- Create `_property_region.py`
- Implement:
  - `NumTargetedTranscriptsProperty`
  - `IsoformConsensusProperty`
- Verify: Region properties calculate correctly

### PHASE 3: Create PropertyCalculator (Steps 11-12)

These steps create the main calculator orchestrator.

**Step 11: Implement PropertyCalculator Class - Part 1 (Basic Structure)**

- Create `_property_calculator.py`
- Implement constructor that takes list of `BaseProperty` instances
- Implement basic `apply()` method (single-threaded for now)
- Test with one simple property (e.g., `GCContentProperty`)
- Verify: Can calculate single property for all oligos in database

**Step 12: Implement PropertyCalculator Class - Part 2 (Parallel Processing)**

- Add parallel processing using `joblib.Parallel`
- Process regions in parallel (similar to `PropertyFilter`)
- Add progress tracking with `joblib_progress`
- Test with multiple properties
- Verify: Parallel processing works correctly, all properties calculated

### PHASE 4: Module Exports and Integration (Steps 13-17)

These steps make the module usable and update dependent code.

**Step 13: Setup Module Exports**

- Complete `__init__.py` with all exports:
  - `BaseProperty`
  - `PropertyCalculator`
  - All property classes from `_property_sequence.py` and `_property_region.py`
  - All calculation functions from `_property_functions.py` (for filters/scorers)
- Verify: All classes and functions can be imported from module

**Step 14: Update Filters to Use New Functions - Part 1**

- Update `_filter_experiment_unspecific.py`:
  - Replace `OligoAttributes._calc_GC_content` with import from `_property_functions`
  - Replace `OligoAttributes._calc_TmNN` with import from `_property_functions`
  - Replace `OligoAttributes._calc_DG_secondary_structure` with import
  - Replace `OligoAttributes._calc_length_complement` with import
- Verify: Filters still work correctly with new imports

**Step 15: Update Filters to Use New Functions - Part 2**

- Update `_filter_experiment_specific.py`:
  - Replace `OligoAttributes()._calc_padlock_arms` with import
  - Replace `OligoAttributes()._calc_detect_oligo` with import
- Verify: Experiment-specific filters work correctly

**Step 16: Update Scorers to Use New Functions**

- Update `_scorer_sequence_property.py`:
  - Replace `OligoAttributes._calc_GC_content` with import
  - Replace `OligoAttributes._calc_TmNN` with import
- Verify: Scorers calculate correctly with new imports

**Step 17: Update Specificity Filters to Use New Functions**

- Find and update any specificity filter files using `OligoAttributes` methods
- Update to import from `_property_functions.py`
- Verify: Specificity filters work correctly

### PHASE 5: Update Pipelines (Steps 18-21)

These steps migrate pipeline code to use the new pattern.

**Step 18: Update One Pipeline File - Test Case**

- Pick one pipeline file (e.g., `_oligo_seq_probe_designer.py`)
- Replace old pattern with new pattern for 1-2 property calculations:
  ```python
  # Old:
  oligo_database = self.oligo_attributes_calculator.calculate_GC_content(...)
  
  # New:
  properties = [GCContentProperty(sequence_type="oligo")]
  calculator = PropertyCalculator(properties=properties)
  oligo_database = calculator.apply(oligo_database=oligo_database, sequence_type="oligo", n_jobs=self.n_jobs)
  ```

- Verify: Pipeline step works correctly with new pattern

**Step 19: Complete Migration of First Pipeline**

- Finish updating the selected pipeline file
- Replace all `oligo_attributes_calculator.calculate_*` calls with new pattern
- Remove `self.oligo_attributes_calculator = OligoAttributes()` initialization
- Verify: Entire pipeline runs successfully

**Step 20: Update Remaining Pipelines**

- Update `_scrinshot_probe_designer.py`
- Update `_merfish_probe_designer.py`
- Update `_seqfish_plus_probe_designer.py`
- Update `_cycle_hcr_probe_designer.py`
- Verify: Each pipeline works after update

**Step 21: Update Specificity Filter Files Using Attribute Calculations**

- Update any specificity filter files that directly call `OligoAttributes` methods
- Replace with `PropertyCalculator` pattern where appropriate
- Verify: All specificity filtering still works

### PHASE 6: Cleanup (Steps 22-24)

These steps remove old code and complete the migration.

**Step 22: Remove Old Implementation**

- Delete `oligo_designer_toolsuite/database/_oligo_database_attributes.py`
- Update `oligo_designer_toolsuite/database/__init__.py` to remove `OligoAttributes` export
- Verify: No remaining imports of `OligoAttributes` class

**Step 23: Update Test Files**

- Update test files to use new `PropertyCalculator` pattern
- Verify: All tests pass

**Step 24: Update Tutorial Notebooks**

- Update tutorial notebooks to use new pattern
- Verify: Tutorials work correctly

## Verification Criteria for Each Step

Each step should:

1. **No breaking changes**: Code should still run (except in final cleanup steps)
2. **Same results**: New code produces identical results to old code
3. **Can be imported**: New modules/classes can be imported without errors
4. **Tests pass**: Existing tests continue to pass (where applicable)

## Key Design Decisions

1. **Always Recalculate**: Properties are always recalculated (no caching checks)
2. **File Organization**: Group properties by type (sequence vs region) following the scorer pattern
3. **Shared Functions**: Calculation logic extracted to `_property_functions.py` for use by both filters and calculators
4. **Parallel Processing**: Use `joblib.Parallel` similar to `PropertyFilter` for parallel region processing
5. **Naming**: Rename "attributes" to "properties" throughout the codebase
6. **Incremental Migration**: Keep old code working until new code is verified

### To-dos

- [ ] Extract all calculation functions from OligoAttributes to _property_functions.py
- [ ] Create BaseProperty abstract base class in _property_base.py
- [ ] Implement sequence-based property classes in _property_sequence.py
- [ ] Implement region-based property classes in _property_region.py
- [ ] Implement PropertyCalculator class with parallel processing in _property_calculator.py
- [ ] Create __init__.py with proper exports for oligo_property_calculator module
- [ ] Update filter classes to import calculation functions from new location
- [ ] Update scorer classes to import calculation functions from new location
- [ ] Update all pipeline files to use new PropertyCalculator pattern
- [ ] Update specificity filter files that use attribute calculations
- [ ] Delete _oligo_database_attributes.py and update database __init__.py
- [ ] Update test files to use new property calculator system
- [ ] Update tutorial notebooks to use new property calculator system