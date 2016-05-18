#!/usr/bin/python
# coding=utf-8
from configparser import ConfigParser
from datetime import datetime

# Database Options
database_dialect = None
database_name = None
database_user = None
database_user_password = None
database_host = 'localhost'
database_port = None
database_eager_load = False

# Logging options
logging_level = 'DEBUG'
logging_file = None
logging_override = True
logging_format = '%(asctime)s [%(levelname)s] %(message)s'
logging_date_format = '%Y.%m.%d %H:%M:%S'

# Reporting options
reporting_display = True
reporting_save = True
reporting_file = 'report'
reporting_target_histogram = False
reporting_validation_curve = False
reporting_learning_curve = False
reporting_display_charts = True
reporting_save_charts = False

# Repository options
repository_name = None

# Dataset options
dataset_cache = False
dataset_cache_dir = None
dataset_target = None
dataset_train_start = None
dataset_train_end = None
dataset_test_start = None
dataset_test_end = None
dataset_ngram_sizes = None
dataset_ngram_levels = None
dataset_features = None

# Machine learning options
ml_model = None
ml_feature_scaling = False
ml_polynomial_degree = 1
ml_alpha = None
ml_alpha_range = None
ml_C = None
ml_C_range = None
ml_cross_validation = None
ml_kernel = 'rbf'


def read_config(config_file):
    """ Reads the config file and initializes the config variables.

    Args:
        config_file (str): The relative filepath to the config file.
    """

    config = ConfigParser()
    config.read(config_file)

    database_section = 'DATABASE'
    _read_option(config, database_section, 'dialect', optional=False)
    _read_option(config, database_section, 'name', optional=False)
    _read_option(config, database_section, 'user')
    _read_option(config, database_section, 'user_password')
    _read_option(config, database_section, 'host')
    _read_option(config, database_section, 'port')
    _read_option(config, database_section, 'eager_load', value_type=TYPE_BOOLEAN)

    logging_section = 'LOGGING'
    _read_option(config, logging_section, 'level')
    _read_option(config, logging_section, 'file')
    _read_option(config, logging_section, 'override', value_type=TYPE_BOOLEAN)
    _read_option(config, logging_section, 'format')
    _read_option(config, logging_section, 'date_format')

    reporting_section = 'REPORTING'
    _read_option(config, reporting_section, 'display_reports', target='reporting_display', value_type=TYPE_BOOLEAN)
    _read_option(config, reporting_section, 'save_reports', target='reporting_save', value_type=TYPE_BOOLEAN)
    _read_option(config, reporting_section, 'file')
    _read_option(config, reporting_section, 'target_histogram', value_type=TYPE_BOOLEAN)
    _read_option(config, reporting_section, 'validation_curve', value_type=TYPE_BOOLEAN)
    _read_option(config, reporting_section, 'learning_curve', value_type=TYPE_BOOLEAN)
    _read_option(config, reporting_section, 'display_charts', value_type=TYPE_BOOLEAN)
    _read_option(config, reporting_section, 'save_charts', value_type=TYPE_BOOLEAN)

    repository_section = 'REPOSITORY'
    _read_option(config, repository_section, 'name', optional=False)

    dataset_section = 'DATASET'
    _read_option(config, dataset_section, 'cache', value_type=TYPE_BOOLEAN)
    _read_option(config, dataset_section, 'cache_dir', value_type=TYPE_BOOLEAN)
    _read_option(config, dataset_section, 'target', optional=False)
    _read_option(config, dataset_section, 'train_start', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'train_end', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'test_start', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'test_end', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'ngram_sizes', value_type=TYPE_INT_LIST)
    _read_option(config, dataset_section, 'ngram_levels', value_type=TYPE_INT_LIST)
    _read_option(config, dataset_section, 'features', optional=False, value_type=TYPE_STR_LIST)

    ml_section = "ML"
    _read_option(config, ml_section, 'model', optional=False)
    _read_option(config, ml_section, 'polynomial_degree', value_type=TYPE_INT)
    _read_option(config, ml_section, 'feature_scaling', value_type=TYPE_BOOLEAN)
    _read_option(config, ml_section, 'alpha', value_type=TYPE_FLOAT)
    _read_option(config, ml_section, 'alpha_range', value_type=TYPE_FLOAT_LIST)
    _read_option(config, ml_section, 'C', target='ml_C', value_type=TYPE_FLOAT)
    _read_option(config, ml_section, 'C_range', target='ml_C_range', value_type=TYPE_FLOAT_LIST)
    _read_option(config, ml_section, 'cross_validation', value_type=TYPE_BOOLEAN)
    _read_option(config, ml_section, 'kernel')


TYPE_STR = 1
TYPE_INT = 2
TYPE_FLOAT = 3
TYPE_BOOLEAN = 4
TYPE_DATE = 5
TYPE_STR_LIST = 6
TYPE_FLOAT_LIST = 7
TYPE_INT_LIST = 8


def _read_option(config, section, option, value_type=TYPE_STR, target=None, optional=True, date_format='%Y.%m.%d'):
    """ Reads an option into a global variable.

    If no target is provided, the global variable is called <section>_<option>.

    Args:
        config (ConfigParser): The ConfigParser object from which to read the option
        section (str): The section name
        option (str): The option name
        value_type (int): Optional. The type. Use one of the TYPE_* constants
        target (str): Optional. The name of the gobal variable to be written
        optional (bool): Optional. If False, an Error will be thrown if the option is not found.
        date_format (str): Optional. The format to convert datestrings to dates. Only used when valuetype is Date.
    """
    if config.has_option(section, option):
        if value_type == TYPE_STR:
            value = config.get(section, option)
        elif value_type == TYPE_INT:
            value = config.getint(section, option)
        elif value_type == TYPE_FLOAT:
            value = config.getfloat(section, option)
        elif value_type == TYPE_BOOLEAN:
            value = config.getboolean(section, option)
        elif value_type == TYPE_DATE:
            value = config.get(section, option)
            value = datetime.strptime(value, date_format)
        elif value_type == TYPE_STR_LIST:
            value = config.get(section, option)
            value = [item.strip() for item in value.split(',')]
        elif value_type == TYPE_FLOAT_LIST:
            value = config.get(section, option)
            value = [float(item.strip()) for item in value.split(',')]
        elif value_type == TYPE_INT_LIST:
            value = config.get(section, option)
            value = [int(item.strip()) for item in value.split(',')]
        else:
            raise ValueError("No valid Type provided")

        if target:
            target_name = target
        else:
            target_name = section.lower() + "_" + option.lower()

        if target_name not in globals().keys():
            raise ValueError("Config has no attribute " + target_name)

        globals()[target_name] = value
    elif not optional:
        if not config.has_section(section):
            raise ConfigError(
                'Obligatory section %s not found in Config file.' % section)
        raise ConfigError("Obligatory option %s not found in section %s" % (option, section))


class ConfigError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
