#!/usr/bin/python
# coding=utf-8
import logging
import argparse
from ml import Dataset, Model, Predict, Scoreboard
from ml import Reporting
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
    logging.info("Starting ML Pipeline!")
    logging.info("Initializing Database")
    try:
        DB.init_db()
    except DBError:
        die("DB Model could not be created!")

    logging.info("Reading training dataset")
    train_dataset = Dataset.get_dataset(
        Config.repository_name,
        Config.dataset_train_start,
        Config.dataset_train_end,
        Config.dataset_features,
        Config.dataset_target,
        label="Training",
        cache=Config.dataset_cache)
    if train_dataset is None:
        die("Training Dataset could not be created!")

    logging.info("Reading test dataset")
    test_dataset = Dataset.get_dataset(
        Config.repository_name,
        Config.dataset_test_start,
        Config.dataset_test_end,
        Config.dataset_features,
        Config.dataset_target,
        label="Test",
        cache=Config.dataset_cache)
    if test_dataset is None:
        die("Test Dataset could not be created!")

    logging.info("Creating and training model with training dataset")
    model = Model.create_model(
        Config.ml_model,
        normalize=Config.ml_normalize,
        alpha=Config.ml_alpha,
    )

    Model.train_model(
        model,
        train_dataset,
        polynomial_degree=Config.ml_polynomial_degree
    )

    logging.info("Model successfully trained.")
    logging.debug("Model coefficients: " + str(model.coef_))

    logging.debug("Creating predictions...")
    baseline_mean_prediction = Predict.predict_mean(train_dataset, test_dataset.target.shape[0])
    baseline_med_prediction = Predict.predict_median(train_dataset, test_dataset.target.shape[0])
    baseline_wr_prediction = Predict.predict_weighted_random(train_dataset, test_dataset.target.shape[0])
    training_prediction = Predict.predict_with_model(
        train_dataset,
        model,
        polynomial_degree=Config.ml_polynomial_degree)
    test_prediction = Predict.predict_with_model(
        test_dataset,
        model,
        polynomial_degree=Config.ml_polynomial_degree)

    logging.debug("Creating reports from predictions")
    baseline_mean_report = Reporting.Report(test_dataset.target, baseline_mean_prediction, "Mean Baseline")
    baseline_med_report = Reporting.Report(test_dataset.target, baseline_med_prediction, "Median Baseline")
    baseline_wr_report = Reporting.Report(test_dataset.target, baseline_wr_prediction, "Weighted Random Baseline")
    training_report = Reporting.Report(train_dataset.target, training_prediction, "Training")
    test_report = Reporting.Report(test_dataset.target, test_prediction, "Test")

    base_entry = Scoreboard.create_entry_from_config(baseline_wr_report)
    test_entry = Scoreboard.create_entry_from_config(test_report)
    Scoreboard.add_entry(base_entry)
    Scoreboard.add_entry(test_entry)
    Scoreboard.write_entries()
    base_ranking = Scoreboard.get_ranking(base_entry, Scoreboard.RATING_ATTRIBUTE_R2S)
    test_ranking = Scoreboard.get_ranking(test_entry, Scoreboard.RATING_ATTRIBUTE_R2S)

    if Config.reporting_display:
        print(baseline_mean_report)
        print(baseline_med_report)
        print(baseline_wr_report)
        print(training_report)
        print(test_report)

        comparisation_table = Reporting.get_report_comparisation_table(
            [baseline_wr_report, training_report, test_report],
            [Reporting.SCORE_R2S, Reporting.SCORE_MAE, Reporting.SCORE_MDE])
        print(comparisation_table.table)

        if Config.ml_polynomial_degree == 1:
            # Determining top features only makes sense without polynomial features.
            top_features_table = Reporting.get_top_features_table(model, train_dataset.feature_list, 5)
            print(top_features_table.table)

        print("Base ranking: %i" % base_ranking)
        print("Test ranking: %i" % test_ranking)

    # TODO: If CV, show learning curve

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
