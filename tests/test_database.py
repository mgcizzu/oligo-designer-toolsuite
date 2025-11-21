############################################
# imports
############################################

import os
import shutil
import unittest

import pandas as pd
import yaml

from oligo_designer_toolsuite.database import OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_property_calculator import (
    DetectOligoProperty,
    DGSecondaryStructureProperty,
    GCContentProperty,
    IsoformConsensusProperty,
    LengthComplementProperty,
    LengthProperty,
    LengthSelfComplementProperty,
    NumTargetedTranscriptsProperty,
    PadlockArmsProperty,
    PropertyCalculator,
    ReverseComplementSequenceProperty,
    SeedregionProperty,
    SeedregionSiteProperty,
    ShortenedSequenceProperty,
    TmNNProperty,
)
from oligo_designer_toolsuite.sequence_generator import OligoSequenceGenerator
from oligo_designer_toolsuite.utils import FastaParser, VCFParser, check_tsv_format

############################################
# setup
############################################

# Global Parameters
FILE_NCBI_EXONS = "tests/data/genomic_regions/sequences_ncbi_exons.fna"
FILE_VARIANTS = "tests/data/annotations/custom_GCF_000001405.40.chr16.vcf"
FILE_DATABASE_OLIGO_PROPERTIES = "tests/data/databases/database_oligo_properties.tsv"

REGION_IDS = [
    "AARS1",
    "DECR2",
    "FAM234A",
    "RHBDF1",
    "WASIR2",
    "this_gene_does_not_exist",
]

############################################
# tests
############################################


class TestReferenceDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_reference_database")

        self.fasta_parser = FastaParser()
        self.vcf_parser = VCFParser()

        self.reference_fasta = ReferenceDatabase(
            database_name="test_reference_database_fasta", dir_output=self.tmp_path
        )
        self.reference_fasta.load_database_from_file(
            files=[FILE_NCBI_EXONS], file_type="fasta", database_overwrite=False
        )
        self.reference_fasta.load_database_from_file(
            files=FILE_NCBI_EXONS, file_type="fasta", database_overwrite=True
        )

        self.reference_vcf = ReferenceDatabase(
            database_name="test_reference_database_vcf", dir_output=self.tmp_path
        )
        self.reference_vcf.load_database_from_file(
            files=FILE_VARIANTS, file_type="vcf", database_overwrite=True
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_path)
        except:
            pass

    def test_write_database(self):
        file_fasta_database = self.reference_fasta.write_database_to_file(filename="ref_db_filtered_fasta")
        assert (
            self.fasta_parser.check_fasta_format(file_fasta_database) == True
        ), f"error: wrong file format for database in {file_fasta_database}"

        file_vcf_database = self.reference_vcf.write_database_to_file(filename="ref_db_filtered_vcf")
        assert (
            self.vcf_parser.check_vcf_format(file_vcf_database) == True
        ), f"error: wrong file format for database in {file_vcf_database}"

    def test_filter_database_by_region(self):
        self.reference_fasta.filter_database_by_region(region_ids="AARS1", keep_region=False)
        fasta_sequences = self.fasta_parser.read_fasta_sequences(
            file_fasta_in=self.reference_fasta.database_file
        )
        for entry in fasta_sequences:
            (
                region,
                _,
                _,
            ) = self.fasta_parser.parse_fasta_header(entry.id)
            assert region != "AARS1", f"error: this region {region} should be filtered out."

    def test_filter_database_by_property_category(self):
        self.reference_fasta.filter_database_by_property_category(
            property_name="gene_id",
            property_category="AARS1",
            keep_if_equals_category=False,
        )
        fasta_sequences = self.fasta_parser.read_fasta_sequences(
            file_fasta_in=self.reference_fasta.database_file
        )
        for entry in fasta_sequences:
            (
                region,
                _,
                _,
            ) = self.fasta_parser.parse_fasta_header(entry.id)
            assert region != "AARS1", f"error: this region {region} should be filtered out."


class TestOligoDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_oligo_database")

        self.fasta_parser = FastaParser()

        self.oligo_sequence_generator = OligoSequenceGenerator(dir_output=self.tmp_path)
        self.oligo_database = OligoDatabase(
            min_oligos_per_region=2,
            write_regions_with_insufficient_oligos=True,
            max_entries_in_memory=10,
            n_jobs=4,
            database_name="test_oligo_database",
            dir_output=self.tmp_path,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def load_database_from_fasta(self):
        file_random_seqs = self.oligo_sequence_generator.create_sequences_random(
            filename_out="random_sequences1",
            length_sequences=30,
            num_sequences=100,
            name_sequences="random_sequences1",
            base_alphabet_with_probability={"A": 0.1, "C": 0.3, "G": 0.4, "T": 0.2},
        )
        file_sliding_window = self.oligo_sequence_generator.create_sequences_sliding_window(
            files_fasta_in=FILE_NCBI_EXONS,
            region_ids=REGION_IDS,
            length_interval_sequences=(30, 31),
            n_jobs=4,
        )
        self.oligo_database.load_database_from_fasta(
            files_fasta=file_random_seqs,
            sequence_type="oligo",
            region_ids="random_sequences1",
            database_overwrite=True,
        )
        self.oligo_database.load_database_from_fasta(
            files_fasta=file_sliding_window,
            sequence_type="target",
            region_ids=REGION_IDS,
            database_overwrite=False,
        )

    def test_load_database_from_fasta(self):
        self.load_database_from_fasta()

        assert len(self.oligo_database.database) == 6, "error: wrong number of sequences loaded into database"

    def test_load_database_from_table(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=["region_1", "region_2"],
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        assert len(self.oligo_database.database) == 2, "error: wrong number of sequences loaded into database"

    def test_load_save_database(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES, database_overwrite=True, merge_databases_on_sequence_type="oligo"
        )

        dir_database = self.oligo_database.save_database(
            region_ids=["region_1", "region_2"], name_database="database_region1_region2"
        )
        self.oligo_database.load_database(
            dir_database, database_overwrite=True, merge_databases_on_sequence_type="oligo"
        )

        assert len(self.oligo_database.database.keys()) == 2, "error: wrong number regions saved and loaded"

    def test_write_database_to_fasta(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=["region_1", "region_2"],
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )
        file_fasta = self.oligo_database.write_database_to_fasta(
            sequence_type="oligo", save_description=True, filename="database_region1_region2"
        )

        assert (
            self.fasta_parser.check_fasta_format(file_fasta) == True
        ), f"error: wrong file format for database in {file_fasta}"

        assert (
            len(self.fasta_parser.get_fasta_regions(file_fasta_in=file_fasta)) == 2
        ), f"error: wrong number of regions stored in {file_fasta}"

    def test_write_database_to_table(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES, database_overwrite=True, merge_databases_on_sequence_type="oligo"
        )

        file_database = self.oligo_database.write_database_to_table(
            properties=["test_property", "ligation_site", "chromosome", "start", "end", "strand"],
            flatten_property=True,
            filename="database_region1_region2_flattened",
            region_ids=["region_1", "region_2"],
        )

        self.oligo_database.load_database_from_table(
            file_database, database_overwrite=True, merge_databases_on_sequence_type="oligo"
        )

        assert check_tsv_format(file_database) == True, f"error: wrong file format"
        assert len(self.oligo_database.database.keys()) == 2, "error: wrong number regions saved and loaded"
        assert (
            self.oligo_database.get_oligo_property_value(
                property="test_property", flatten=True, region_id="region_1", oligo_id="region_1::1"
            )
            == "red"
        ), f"error: wrong property stored in {file_database}"

        file_database = self.oligo_database.write_database_to_table(
            properties=["test_property", "ligation_site", "chromosome", "start", "end", "strand"],
            flatten_property=False,
            filename="database_region1_region2_unflattened",
            region_ids=["region_1", "region_2"],
        )

        self.oligo_database.load_database_from_table(
            file_database, database_overwrite=True, merge_databases_on_sequence_type="oligo"
        )

        assert check_tsv_format(file_database) == True, f"error: wrong file format"
        assert len(self.oligo_database.database.keys()) == 2, "error: wrong number regions saved and loaded"
        assert (
            self.oligo_database.get_oligo_property_value(
                property="test_property", flatten=True, region_id="region_1", oligo_id="region_1::1"
            )
            == "red"
        ), f"error: wrong property stored in {file_database}"

    def test_write_database_to_bed(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=["region_1", "region_2"],
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        file_bed = self.oligo_database.write_database_to_bed(
            filename="database_region1_region2_bed", region_ids=["region_1", "region_2"]
        )

        assert check_tsv_format(file_bed) == True, f"error: wrong file format"

        bed_table = pd.read_csv(
            file_bed, sep="\t", names=["chromosome", "start", "end", "name", "score", "strand"]
        )

        assert bed_table.loc[0, "start"] == 70289456
        assert bed_table.loc[0, "end"] == 70289485

    def test_write_oligosets_to_yaml(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES,
            database_overwrite=True,
            region_ids="region_1",
            merge_databases_on_sequence_type="oligo",
        )

        oligoset = pd.DataFrame(
            data=[
                [0, "region_1::1", "region_1::2", "region_1::5", 1.59, 2.36],
                [1, "region_1::6", "region_1::4", "region_1::9", 2.15, 4.93],
            ],
            columns=[
                "oligoset_id",
                "oligo_0",
                "oligo_1",
                "oligo_2",
                "set_score_lowest",
                "set_score_sum",
            ],
        )

        self.oligo_database.oligosets["region_1"] = oligoset

        file_yaml = self.oligo_database.write_oligosets_to_yaml(
            properties=[
                "test_property",
                "ligation_site",
                "chromosome",
                "start",
                "end",
                "strand",
                "transcript_id",
            ],
            top_n_sets=2,
            ascending=True,
        )

        with open(file_yaml, "r") as handle:
            yaml_oligosets = yaml.safe_load(handle)

        assert yaml_oligosets["region_1"]["Oligoset 1"]["Oligoset Score"] == {
            "set_score_lowest": 1.59,
            "set_score_sum": 2.36,
        }, f"error: wrong oligoset loaded"

        assert yaml_oligosets["region_1"]["Oligoset 1"]["Oligo 1"]["test_property"] == [
            "red"
        ], f"error: wrong oligoset loaded"

        assert yaml_oligosets["region_1"]["Oligoset 1"]["Oligo 1"]["transcript_id"] == [
            ["NM_001605.3"],
            ["XM_047433666.1"],
        ], f"error: wrong oligoset loaded"

    def test_write_oligosets_to_table(self):
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES,
            database_overwrite=True,
            region_ids="region_1",
            merge_databases_on_sequence_type="oligo",
        )

        oligoset = pd.DataFrame(
            data=[
                [0, "region_1::1", "region_1::2", "region_1::5", 1.59, 2.36],
                [1, "region_1::6", "region_1::4", "region_1::9", 2.15, 4.93],
            ],
            columns=[
                "oligoset_id",
                "oligo_0",
                "oligo_1",
                "oligo_2",
                "set_score_lowest",
                "set_score_sum",
            ],
        )

        self.oligo_database.oligosets["region_1"] = oligoset

        folder_oligosets = self.oligo_database.write_oligosets_to_table()
        file_oligosets = os.path.join(folder_oligosets, "oligosets_region_1.tsv")
        assert (
            check_tsv_format(file=file_oligosets) == True
        ), f"error: incorrect file format of {file_oligosets}"

    def test_remove_regions_with_insufficient_oligos(self):
        self.load_database_from_fasta()
        self.oligo_database.remove_regions_with_insufficient_oligos("database_generation")
        assert len(self.oligo_database.database.keys()) == (
            len(REGION_IDS) - 1 + 1  # one region removed but one added from random seqs
        ), "error: wrong number of regions in database"

    def test_get_property_list(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        list_properties = self.oligo_database.get_property_list()

        assert len(list_properties) == 13, "error: wrong number of properties in database"
        assert "oligo" in list_properties, "error: missing property"

    def test_get_regionid_list(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        list_regionid = self.oligo_database.get_regionid_list()
        assert len(list_regionid) == 3, "error: wrong number of regionids in database"
        assert "region_3" in list_regionid, "error: missing regionid"

    def test_get_oligoid_list(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        list_oligoids = self.oligo_database.get_oligoid_list()
        assert len(list_oligoids) == 21, "error: wrong number of oligoids in database"
        assert "region_3::1" in list_oligoids, "error: missing oligoid"

    def test_get_sequence_list(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        list_sequences = self.oligo_database.get_sequence_list(sequence_type="oligo")
        assert len(list_sequences) == 21, "error: wrong number of sequences in database"
        assert "TATAACCCTGAGGAGGTATACCTAG" in list_sequences, "error: missing sequence"

    def test_get_oligoid_sequence_mapping(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        mapping = self.oligo_database.get_oligoid_sequence_mapping(sequence_type="oligo")
        assert mapping["region_1::1"] == "ATGCCCCAATGGATGACGAT", "error: wrong sequence for oligoid"
        assert mapping["region_3::5"] == "CTCACTCGACTCTTACACAGTCATA", "error: wrong sequence for oligoid"

        mapping = self.oligo_database.get_oligoid_sequence_mapping(sequence_type="target")
        assert mapping["region_1::1"] == "ATCGTCATCCATTGGGGCAT", "error: wrong sequence for oligoid"
        assert mapping["region_3::5"] == "TATGACTGTGTAAGAGTCGAGTGAG", "error: wrong sequence for oligoid"

    def test_get_sequence_oligoid_mapping(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        mapping = self.oligo_database.get_sequence_oligoid_mapping(sequence_type="oligo")
        assert len(mapping["CTCACTCGACTCTTACACAGTCATA"]) == 4, "error: wrong number of oligos for sequence"

    def test_get_oligo_property_table(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )
        property_table = self.oligo_database.get_oligo_property_table(
            properties="test_property", flatten=True
        )

        assert (
            len(property_table.explode("test_property")["test_property"].unique()) == 2
        ), "error: wrong property returned"

    def test_get_oligo_property_value(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )
        property1 = self.oligo_database.get_oligo_property_value(
            property="test_property", flatten=True, region_id="region_1", oligo_id="region_1::5"
        )
        property2 = self.oligo_database.get_oligo_property_value(
            property="test_property", flatten=False, region_id="region_3", oligo_id="region_3::3"
        )

        assert property1 == "red", "error: wrong property value returned"
        assert property2 == [["blue"]], "error: wrong property value returned"

    def test_update_oligo_properties(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids="region_3",
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )
        new_property = {
            "region_3::1": {"GC_content": 63},
            "region_3::2": {"GC_content": 66},
            "region_3::3": {"GC_content": 80},
            "region_3::4": {"GC_content": 70},
            "region_3::5": {"GC_content": 40},
        }
        self.oligo_database.update_oligo_properties(new_oligo_property=new_property)
        property_table = self.oligo_database.get_oligo_property_table(properties="GC_content", flatten=True)

        assert len(property_table) == 5, "error: property not correctly updated"

    def test_filter_database_by_region(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        self.oligo_database.filter_database_by_region(remove_region=True, region_ids="region_3")

        assert len(self.oligo_database.database.keys()) == 2, "error: remove region was kept"

        self.oligo_database.filter_database_by_region(
            remove_region=False, region_ids=["region_1", "region_2"]
        )

        assert len(self.oligo_database.database.keys()) == 2, "error: keep regions were removed"

    def test_filter_database_by_oligo(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        self.oligo_database.filter_database_by_oligo(remove_region=True, oligo_ids="region_3::5")

        assert len(self.oligo_database.database["region_3"].keys()) == 4, "error: remove oligo was kept"

        self.oligo_database.filter_database_by_oligo(
            remove_region=False, oligo_ids=["region_3::3", "region_3::4"]
        )

        assert len(self.oligo_database.database["region_3"].keys()) == 2, "error: keep oligo were removed"

    def test_filter_database_by_property_threshold(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids="region_3",
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )
        new_property = {
            "region_3::1": {"GC_content": 63},
            "region_3::2": {"GC_content": 66},
            "region_3::3": {"GC_content": 80},
            "region_3::4": {"GC_content": 70},
            "region_3::5": {"GC_content": 40},
        }
        self.oligo_database.update_oligo_properties(new_oligo_property=new_property)

        self.oligo_database.filter_database_by_property_threshold(
            property_name="GC_content", property_thr=65, remove_if_smaller_threshold=True
        )
        assert len(self.oligo_database.get_oligoid_list()) == 3, "error: wrong number of oligos filtered"

        self.oligo_database.filter_database_by_property_threshold(
            property_name="GC_content", property_thr=75, remove_if_smaller_threshold=False
        )
        assert len(self.oligo_database.get_oligoid_list()) == 2, "error: wrong number of oligos filtered"

    def test_filter_database_by_property_category(self):
        self.oligo_database.load_database_from_table(
            file_database=FILE_DATABASE_OLIGO_PROPERTIES,
            region_ids=None,
            database_overwrite=True,
            merge_databases_on_sequence_type="oligo",
        )

        self.oligo_database.filter_database_by_property_category(
            property_name="test_property", property_category="red", remove_if_equals_category=True
        )
        assert len(self.oligo_database.get_oligoid_list()) == 9, "error: wrong number of oligos filtered"


class TestOligoProperties(unittest.TestCase):
    def setUp(self):
        self.tmp_path = os.path.join(os.getcwd(), "tmp_oligo_properties")

        self.oligo_database = OligoDatabase(
            min_oligos_per_region=2,
            write_regions_with_insufficient_oligos=True,
            database_name="test_oligo_properties",
            dir_output=self.tmp_path,
        )
        self.oligo_database.load_database_from_table(
            FILE_DATABASE_OLIGO_PROPERTIES, database_overwrite=True, merge_databases_on_sequence_type="oligo"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_calculate_oligo_length(self):
        properties = [LengthProperty()]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        length1 = oligo_database.get_oligo_property_value(
            property="length_oligo", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )
        length2 = oligo_database.get_oligo_property_value(
            property="length_oligo", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )

        assert length1 == 20, "error: wrong oligo length"
        assert length2 == 29, "error: wrong oligo length"

    def test_calculate_reverse_complement_sequence(self):
        properties = [ReverseComplementSequenceProperty(sequence_type_reverse_complement="oligo")]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(
            oligo_database=self.oligo_database, sequence_type="target", n_jobs=1
        )

        target = oligo_database.get_oligo_property_value(
            property="target", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )
        oligo = oligo_database.get_oligo_property_value(
            property="oligo", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )

        assert target == "ATCGTCATCCATTGGGGCAT", "error: wrong target sequence"
        assert oligo == "ATGCCCCAATGGATGACGAT", "error: wrong oligo sequence"

    def test_calculate_shortened_sequence(self):
        # check if it works for oligos
        sequence_type = "oligo"

        properties = [ShortenedSequenceProperty(sequence_length=10, reverse=False)]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(
            oligo_database=self.oligo_database, sequence_type=sequence_type, n_jobs=1
        )

        sequence_short_oligo = oligo_database.get_oligo_property_value(
            property=f"{sequence_type}_short", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )

        properties = [ShortenedSequenceProperty(sequence_length=10, reverse=True)]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(
            oligo_database=self.oligo_database, sequence_type=sequence_type, n_jobs=1
        )

        sequence_short_oligo_reverse_read = oligo_database.get_oligo_property_value(
            property=f"{sequence_type}_short", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )

        # check if it works for targets
        sequence_type = "target"

        properties = [ShortenedSequenceProperty(sequence_length=5, reverse=False)]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(
            oligo_database=self.oligo_database, sequence_type=sequence_type, n_jobs=1
        )

        sequence_short_target = oligo_database.get_oligo_property_value(
            property=f"{sequence_type}_short", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )

        properties = [ShortenedSequenceProperty(sequence_length=5, reverse=True)]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(
            oligo_database=self.oligo_database, sequence_type=sequence_type, n_jobs=1
        )

        sequence_short_target_reverse_read = oligo_database.get_oligo_property_value(
            property=f"{sequence_type}_short", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )
        assert sequence_short_oligo == "GGCTAGGGAA", "error: wrong short sequence calculated"
        assert sequence_short_target == "CTCTA", "error: wrong short sequence calculated"
        assert sequence_short_oligo_reverse_read == "TCCAATAGAG", "error: wrong short sequence calculated"
        assert sequence_short_target_reverse_read == "TAGCC", "error: wrong short sequence calculated"

    def test_calculate_num_targeted_transcripts(self):
        properties = [NumTargetedTranscriptsProperty()]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        num_targeted_transcripts1 = oligo_database.get_oligo_property_value(
            property="num_targeted_transcripts", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )
        num_targeted_transcripts2 = oligo_database.get_oligo_property_value(
            property="num_targeted_transcripts", flatten=True, region_id="region_2", oligo_id="region_2::1"
        )
        num_targeted_transcripts3 = oligo_database.get_oligo_property_value(
            property="num_targeted_transcripts", flatten=True, region_id="region_3", oligo_id="region_3::4"
        )

        assert num_targeted_transcripts1 == 2, "error: wrong number targeted transcripts"
        assert num_targeted_transcripts2 == 1, "error: wrong number targeted transcripts"
        assert num_targeted_transcripts3 == 28, "error: wrong number targeted transcripts"

    def test_calculate_isoform_consensus(self):
        properties = [NumTargetedTranscriptsProperty()]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)
        properties = [IsoformConsensusProperty()]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=oligo_database, sequence_type="oligo", n_jobs=1)

        isoform_consensus1 = oligo_database.get_oligo_property_value(
            property="isoform_consensus", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )
        isoform_consensus2 = oligo_database.get_oligo_property_value(
            property="isoform_consensus", flatten=True, region_id="region_2", oligo_id="region_2::1"
        )

        assert isoform_consensus1 == 100, "error: wrong isoform consensus, should be 100%"
        assert isoform_consensus2 == 50, "error: wrong isoform consensus, should be 50%"

    def test_calculate_seedregion(self):
        properties = [SeedregionProperty(start=0.4, end=0.6)]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        seedregion_start = oligo_database.get_oligo_property_value(
            property="seedregion_start", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )
        seedregion_end = oligo_database.get_oligo_property_value(
            property="seedregion_end", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )

        assert (seedregion_start == 8) and (seedregion_end == 12), "error: wrong seedregion calculated"

    def test_calculate_seedregion_site(self):
        properties = [SeedregionSiteProperty(seedregion_size=5, seedregion_site_name="ligation_site")]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        seedregion_start = oligo_database.get_oligo_property_value(
            property="seedregion_start", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )
        seedregion_end = oligo_database.get_oligo_property_value(
            property="seedregion_end", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )

        assert (seedregion_start == 6) and (seedregion_end == 15), "error: wrong seedregion calculated"

    def test_calculate_GC_content(self):
        properties = [GCContentProperty()]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        GC_content = oligo_database.get_oligo_property_value(
            property="GC_content_oligo", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )

        assert GC_content == 50, "error: wrong GC content calculated"

    def test_calculate_TmNN(self):
        properties = [TmNNProperty(Tm_parameters={})]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        TmNN = oligo_database.get_oligo_property_value(
            property="TmNN_oligo", flatten=True, region_id="region_1", oligo_id="region_1::1"
        )

        assert TmNN == 53.57, "error: wrong Tm calculated"

    def test_calculate_length_selfcomplement(self):
        properties = [LengthSelfComplementProperty()]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        length_selfcomplement = oligo_database.get_oligo_property_value(
            property="length_selfcomplement_oligo",
            flatten=True,
            region_id="region_3",
            oligo_id="region_3::3",
        )

        assert length_selfcomplement == 18, "error: wrong length of selfcomplement calculated"

    def test_calculate_length_complement(self):
        properties = [LengthComplementProperty(comparison_sequence="AGTC")]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        length_complement = oligo_database.get_oligo_property_value(
            property="length_complement_oligo_AGTC",
            flatten=True,
            region_id="region_1",
            oligo_id="region_1::3",
        )

        assert length_complement == 3, "error: wrong length of complement calculated"

    def test_calculate_secondary_structure_DG(self):
        properties = [DGSecondaryStructureProperty(T=37)]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        DG_secondary_structure = oligo_database.get_oligo_property_value(
            property="DG_secondary_structure_oligo",
            flatten=True,
            region_id="region_1",
            oligo_id="region_1::1",
        )

        assert DG_secondary_structure == 0.8, "error: wrong DG calculated"

    def test_calculate_padlock_arms(self):
        properties = [
            PadlockArmsProperty(
                arm_length_min=3,
                arm_Tm_dif_max=15,
                arm_Tm_min=30,
                arm_Tm_max=80,
                Tm_parameters={},
            )
        ]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        ligation_site = oligo_database.get_oligo_property_value(
            property="ligation_site", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )

        assert ligation_site == 14, "error: wrong padlock arms calculated"

    def test_calculate_detect_oligo(self):
        properties = [
            DetectOligoProperty(
                detect_oligo_length_min=8,
                detect_oligo_length_max=12,
                min_thymines=2,
            )
        ]
        calculator = PropertyCalculator(properties=properties)
        oligo_database = calculator.apply(oligo_database=self.oligo_database, sequence_type="oligo", n_jobs=1)

        detect_oligo_even = oligo_database.get_oligo_property_value(
            property="detect_oligo_even", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )
        detect_oligo_long_left = oligo_database.get_oligo_property_value(
            property="detect_oligo_long_left", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )
        detect_oligo_long_right = oligo_database.get_oligo_property_value(
            property="detect_oligo_long_right", flatten=True, region_id="region_1", oligo_id="region_1::2"
        )

        assert detect_oligo_even == "AGGGAATCGAAT", "error: wrong detection oligo even calculated"
        assert detect_oligo_long_left == None, "error: wrong detection oligo left calculated"
        assert detect_oligo_long_right == None, "error: wrong detection oligo right calculated"
