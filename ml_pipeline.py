#!/usr/bin/python
# coding=utf-8
import logging
import argparse
import datetime
import platform

if platform.system() == 'Linux':
    import matplotlib

    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')

from ml import Dataset, Model, Predict, Scoreboard, LogTransform
from ml import Reporting
from model import DB
from model.DB import DBError
from utils import Config
from utils.Config import ConfigError

report_str = ""


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
        ngram_sizes=Config.dataset_ngram_sizes,
        ngram_levels=Config.dataset_ngram_levels,
        label="Training",
        cache=Config.dataset_cache,
        eager_load=Config.database_eager_load,
        sparse=Config.dataset_sparse
    )
    if train_dataset is None:
        die("Training Dataset could not be created!")
    if Config.ml_log_transform_target:
        train_dataset.target = LogTransform.log_transform(train_dataset.target, base=Config.ml_log_transform_base)

    logging.info("Reading test dataset")
    test_dataset = Dataset.get_dataset(
        Config.repository_name,
        Config.dataset_test_start,
        Config.dataset_test_end,
        Config.dataset_features,
        Config.dataset_target,
        ngram_sizes=Config.dataset_ngram_sizes,
        ngram_levels=Config.dataset_ngram_levels,
        label="Test",
        cache=Config.dataset_cache,
        eager_load=Config.database_eager_load,
        sparse=Config.dataset_sparse
    )
    if test_dataset is None:
        die("Test Dataset could not be created!")
    if Config.ml_log_transform_target:
        test_dataset.target = LogTransform.log_transform(test_dataset.target, base=Config.ml_log_transform_base)

    logging.info("Creating and training model with training dataset")
    model = Model.create_model(
        Config.ml_model,
        feature_scaling=Config.ml_feature_scaling,
        polynomial_degree=Config.ml_polynomial_degree,
        cross_validation=Config.ml_cross_validation,
        alpha=Config.ml_alpha,
        C=Config.ml_C,
        kernel=Config.ml_kernel,
        svr_degree=Config.ml_svr_degree,
        svr_epsilon=Config.ml_svr_epsilon,
        svr_gamma=Config.ml_svr_gamma,
        svr_coef0=Config.ml_svr_coef0,
        sparse=Config.dataset_sparse
    )

    Model.train_model(
        model,
        train_dataset
    )

    logging.info("Model successfully trained.")

    logging.debug("Creating predictions...")
    baseline_mean_prediction = Predict.predict_mean(train_dataset, test_dataset.target.shape[0])
    baseline_med_prediction = Predict.predict_median(train_dataset, test_dataset.target.shape[0])
    baseline_wr_prediction = Predict.predict_weighted_random(train_dataset, test_dataset.target.shape[0])
    training_prediction = Predict.predict_with_model(
        train_dataset,
        model)
    test_prediction = Predict.predict_with_model(
        test_dataset,
        model)

    logging.debug("Creating reports from predictions")

    train_target = train_dataset.target
    test_target = test_dataset.target
    if Config.ml_log_transform_target:
        train_target = LogTransform.exp_transform(train_target, Config.ml_log_transform_base)
        training_prediction = LogTransform.exp_transform(training_prediction, Config.ml_log_transform_base)
        test_target = LogTransform.exp_transform(test_target, Config.ml_log_transform_base)
        test_prediction = LogTransform.exp_transform(test_prediction, Config.ml_log_transform_base)
        baseline_mean_prediction = LogTransform.exp_transform(baseline_mean_prediction, Config.ml_log_transform_base)
        baseline_med_prediction = LogTransform.exp_transform(baseline_med_prediction, Config.ml_log_transform_base)
        baseline_wr_prediction = LogTransform.exp_transform(baseline_wr_prediction, Config.ml_log_transform_base)

    baseline_mean_report = Reporting.Report(test_target, baseline_mean_prediction, "Mean Baseline")
    baseline_med_report = Reporting.Report(test_target, baseline_med_prediction, "Median Baseline")
    baseline_wr_report = Reporting.Report(test_target, baseline_wr_prediction, "Weighted Random Baseline")
    training_report = Reporting.Report(train_target, training_prediction, "Training")
    test_report = Reporting.Report(test_target, test_prediction, "Test")

    base_entry = Scoreboard.create_entry_from_config(baseline_wr_report)
    test_entry = Scoreboard.create_entry_from_config(test_report)
    Scoreboard.add_entry(base_entry)
    Scoreboard.add_entry(test_entry)
    Scoreboard.write_entries()
    base_ranking = Scoreboard.get_ranking(base_entry, Scoreboard.RATING_ATTRIBUTE_R2S)
    test_ranking = Scoreboard.get_ranking(test_entry, Scoreboard.RATING_ATTRIBUTE_R2S)

    if Config.reporting_display or Config.reporting_save:
        config_table = Reporting.get_config_table()
        add_to_report(config_table.table)

        add_to_report(baseline_mean_report)
        add_to_report(baseline_med_report)
        add_to_report(baseline_wr_report)
        add_to_report(training_report)
        add_to_report(test_report)

        comparisation_table = Reporting.get_report_comparisation_table(
            [baseline_wr_report, training_report, test_report],
            [Reporting.SCORE_R2S, Reporting.SCORE_MAE, Reporting.SCORE_MDE])
        add_to_report(comparisation_table.table)

        category_table = Reporting.get_category_table(
            train_target, training_prediction, label="Training prediction")
        add_to_report(category_table.table)

        category_table = Reporting.get_category_table(
            test_target, test_prediction, label="Test prediction")
        add_to_report(category_table.table)

        confusion_matrix_table, classification_report = Reporting.get_confusion_matrix(test_target, test_prediction,
                                                                                       label="Training prediction")
        add_to_report(confusion_matrix_table.table)
        add_to_report(classification_report)
        confusion_matrix_table, classification_report = Reporting.get_confusion_matrix(test_target, test_prediction,
                                                                                       label="Test prediction")
        add_to_report(confusion_matrix_table.table)
        add_to_report(classification_report)

        if Config.ml_polynomial_degree == 1:
            # Determining top features only makes sense without polynomial features.
            top_features_table = Reporting.get_top_features_table(model, train_dataset.feature_list, 10)
            if top_features_table is not None:
                add_to_report(top_features_table.table)

        add_to_report("Base ranking: %i" % base_ranking)
        add_to_report("Test ranking: %i" % test_ranking)
        if test_ranking == 0:
            add_to_report("Congratulations! Best one so far!")
        elif base_ranking > test_ranking:
            add_to_report("Hey, at least better than the baseline!")
        else:
            add_to_report("Do you even learn?")

        if Config.reporting_display:
            print(report_str)

        if Config.reporting_save:
            Reporting.save_report_file(report_str, filename=Config.reporting_file)

        if Config.reporting_target_histogram:
            Reporting.plot_target_histogram(
                train_dataset,
                display=Config.reporting_display_charts,
                save=Config.reporting_save_charts,
            )

        if Config.reporting_validation_curve and Config.ml_cross_validation:
            Reporting.plot_validation_curve(
                model_type=Config.ml_model,
                train_dataset=train_dataset,
                estimator=model,
                alpha=Config.ml_alpha,
                C=Config.ml_C,
                feature_scaling=Config.ml_feature_scaling,
                polynomial_degree=Config.ml_polynomial_degree,
                kernel=Config.ml_kernel,
                svr_degree=Config.ml_svr_degree,
                svr_epsilon=Config.ml_svr_epsilon,
                svr_gamma=Config.ml_svr_gamma,
                svr_coef0=Config.ml_svr_coef0,
                sparse=Config.dataset_sparse,
                display=Config.reporting_display_charts,
                save=Config.reporting_save_charts
            )

        if Config.reporting_learning_curve:
            Reporting.plot_learning_curve(
                train_dataset=train_dataset,
                estimator=model,
                display=Config.reporting_display_charts,
                save=Config.reporting_save_charts
            )

    logging.info("All done. Exiting ML Pipeline")


def add_to_report(string, line_breaks=2):
    global report_str
    report_str += "\n" * line_breaks + str(string)


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


def init_logging(log_file=None, log_level=None, log_override=None, log_format=None, log_date_format=None,
                 log_file_timestamp_format="%Y_%m_%d_%H_%M"):
    """ Initializes the root logger. If Arguments are not provided, they will be read from the Config

    Args:
        log_file (str): Optional. Path to the logging file. If None, logs will only be written to console.
        log_level (str): Optional. The minimum level to be logged. Use 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'
        log_override (bool): Optional. If True the log file will overwritten for each run. Otherwise, logs get appended.
        log_format (str): Optional. The format string fo the log messages.
        log_date_format (str): Optional. The format string for the time info in log messages.
        log_file_timestamp_format (str): Optional. The format string for the timestamp in the log file name.
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

    if not log_override:
        file_mode = 'w'
    else:
        file_mode = 'a'

    log_formatter = logging.Formatter(fmt=log_format, datefmt=log_date_format)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)

    if log_file is not None:
        if not log_override:
            time_str = datetime.datetime.now().strftime(log_file_timestamp_format)
            log_file += "_" + time_str
        file_handler = logging.FileHandler(log_file + ".log", mode=file_mode)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(numeric_log_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(numeric_log_level)
    root_logger.addHandler(console_handler)


if __name__ == '__main__':
    main()
