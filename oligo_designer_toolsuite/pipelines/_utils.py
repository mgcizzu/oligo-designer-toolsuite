############################################
# imports
############################################

import inspect
import logging
import os
import sys
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime

from oligo_designer_toolsuite.database import OligoDatabase

############################################
# Utils functions
############################################


def setup_logging(
    dir_output: str,
    pipeline_name: str,
    log_level: int = logging.NOTSET,
    include_console: bool = False,
    log_start_message: bool = False,
) -> None:
    """
    Set up logging configuration for a pipeline.

    This function creates a consistent logging setup across all pipelines, creating a log file
    in the output directory and optionally writing to the console.

    :param dir_output: Directory path where output files will be saved.
    :type dir_output: str
    :param pipeline_name: Name of the pipeline (used in log file name).
    :type pipeline_name: str
    :param log_level: Logging level (default: logging.NOTSET).
    :type log_level: int
    :param include_console: Whether to also log to console (default: False).
    :type include_console: bool
    :param log_start_message: Whether to log a "START PIPELINE" message (default: False).
    :type log_start_message: bool
    """
    timestamp = datetime.now()
    file_logger = os.path.join(
        dir_output,
        f"log_{pipeline_name}_{timestamp.year}-{timestamp.month}-{timestamp.day}-{timestamp.hour}-{timestamp.minute}.txt",
    )

    handlers = [logging.FileHandler(file_logger)]
    if include_console:
        handlers.append(logging.StreamHandler())

    logging.getLogger("log_name")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=log_level,
        handlers=handlers,
        force=True,  # Force reconfiguration if logging was already configured
    )
    logging.captureWarnings(True)

    if log_start_message:
        logging.info("--------------START PIPELINE--------------")


def base_parser():
    parser = ArgumentParser(
        prog="Genomic Region Generator",
        usage="genomic_region_generation [options]",
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file in yaml format, str",
        default=None,
        type=str,
        metavar="",
    )
    args = parser.parse_args()
    return vars(args)


def base_log_parameters(parameters):

    for key, value in parameters.items():
        if key != "self":
            logging.info("Parameter: %s = %s", key, value)


def log_parameters_and_get_db(func, args, kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    logging.info("Function: %s", func.__name__)
    for name, value in bound_args.arguments.items():
        if name != "self":
            logging.info("Parameter: %s = %s", name, value)

    return bound_args.arguments.get("oligo_database")


def get_oligo_database_info(oligo_database: dict):

    num_genes = len(oligo_database)
    num_oligos = sum(len(oligos) for oligos in oligo_database.values())
    return num_genes, num_oligos


def get_oligo_length_min_max_from_database(oligo_database: dict):

    oligo_length_min = sys.maxsize
    oligo_length_max = 0

    for region in oligo_database.keys():
        for oligo in oligo_database[region].keys():
            length = oligo_database[region][oligo]["length"]
            if length < oligo_length_min:
                oligo_length_min = length
            if length > oligo_length_max:
                oligo_length_max = length

    return oligo_length_min, oligo_length_max


def pipeline_step_basic(step_name: str):
    def decorator(function):
        def wrapper(*args, **kwargs):
            logging.info(f"Parameters {step_name}:")
            log_parameters_and_get_db(function, args, kwargs)

            oligo_database = function(*args, **kwargs)

            num_genes, num_oligos = get_oligo_database_info(oligo_database.database)
            logging.info(
                f"Step - {step_name}: database contains {num_oligos} oligos from {num_genes} regions."
            )

            return oligo_database

        return wrapper

    return decorator


def pipeline_step_advanced(step_name: str):
    def decorator(function):
        def wrapper(*args, **kwargs):
            logging.info(f"Parameters {step_name}:")
            oligo_database = log_parameters_and_get_db(function, args, kwargs)

            num_genes_before, num_oligos_before = get_oligo_database_info(oligo_database.database)

            oligo_database, *returned_values = function(*args, **kwargs)

            num_genes_after, num_oligos_after = get_oligo_database_info(oligo_database.database)
            logging.info(
                f"Step - {step_name}: database contains {num_oligos_after} oligos from {num_genes_after} regions, "
                f"{num_oligos_before - num_oligos_after} oligos and {num_genes_before - num_genes_after} regions removed."
            )

            return oligo_database, *returned_values

        return wrapper

    return decorator


def check_content_oligo_database(oligo_database: OligoDatabase):

    if len(oligo_database.get_regionid_list()) == 0:
        print("The oligo database is empty. Exiting program...")
        warnings.warn("The oligo database is empty. Exiting program...", UserWarning)
        sys.exit(1)  # Exit the program with a status code of 1
