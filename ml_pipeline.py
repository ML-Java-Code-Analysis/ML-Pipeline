#!/usr/bin/python
# coding=utf-8
import logging
import argparse

from ml import Dataset, Learner
from model import DB
from model.DB import DBError
from utils import Config
from utils.Config import ConfigError


def main():
    cli_args = parse_arguments()
    try:
        Config.read_config(cli_args.config_file)
    except ConfigError:
        die("Config File %s could not be read correctly! " % cli_args.config_file)
    init_logging()
    try:
        DB.init_db()
    except DBError:
        die("DB Model could not be created!")

    # TODO: Get all(?) Versions in a certain range from the DB (maybe separate learn/test)
    train_dataset = Dataset.get_dataset_from_range(
        Config.repository_name,
        Config.dataset_train_start,
        Config.dataset_train_end)
    if train_dataset is None:
        die("Training Dataset could not be created!")

    test_dataset = Dataset.get_dataset_from_range(
        Config.repository_name,
        Config.dataset_test_start,
        Config.dataset_test_end)
    if test_dataset is None:
        die("Test Dataset could not be created!")

    model = Learner.train_model(Config.ml_model, train_dataset)
    print(model.coef_)

    # TODO: Test on the Test Set
    # TODO: Print score, some useful analytics and some fancy charts
    # TODO: Maybe save classifier?

    logging.info("All done. Exiting ML Pipeline")


def die(message=None):
    if message:
        logging.critical(message)
    logging.critical("Something went horribly wrong. Exiting ML Pipeline")
    exit()


def parse_arguments():
    """ Parses the provided command line arguments.

    Returns:
        A collection of argument values.
    """
    parser = argparse.ArgumentParser(description='ML-Pipeline')
    parser.add_argument(
        '-c',
        action="store",
        required=False,
        dest="config_file",
        default="ml_pipeline.config",
        help="The path to the config file.")
    return parser.parse_args()


def init_logging(log_file=None, log_level=None, log_override=True, log_format=None, log_date_format=None):
    """ Initializes the root logger. If Arguments are not provided, they will be read from the Config

    Args:
        log_file (str): Optional. Path to the logging file. If None, logs will only be written to console.
        log_level (str): Optional. The minimum level to be logged. Use 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'
        log_override (bool): Optional. If True the log file will overwritten for each run. Otherwise, logs get appended.
        log_format (str): Optional. The format string fo the log messages.
        log_date_format (str): Optional. The format string for the time info in log messages.
    """
    if not log_file:
        log_file = Config.logging_file
    if not log_level:
        log_level = Config.logging_level
    if not log_override:
        log_override = Config.logging_override
    if not log_format:
        log_format = Config.logging_format
    if not log_date_format:
        log_date_format = Config.logging_date_format

    numeric_log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    file_mode = 'w' if log_override else None

    log_formatter = logging.Formatter(fmt=log_format, datefmt=log_date_format)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(numeric_log_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(numeric_log_level)
    root_logger.addHandler(console_handler)


main()
