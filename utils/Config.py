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

# Logging options
logging_level = 'DEBUG'
logging_file = None
logging_override = True
logging_format = '%(asctime)s [%(levelname)s] %(message)s'
logging_date_format = '%Y.%m.%d %H:%M:%S'

# Repository options
repository_name = None

# Dataset options
dataset_target = None
dataset_train_start = None
dataset_train_end = None
dataset_test_start = None
dataset_test_end = None
dataset_features = None

# Machine learning options
ml_model = None


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

    logging_section = 'LOGGING'
    _read_option(config, logging_section, 'level')
    _read_option(config, logging_section, 'file')
    _read_option(config, logging_section, 'override', value_type=TYPE_BOOLEAN)
    _read_option(config, logging_section, 'format')
    _read_option(config, logging_section, 'date_format')

    repository_section = 'REPOSITORY'
    _read_option(config, repository_section, 'name', optional=False)

    dataset_section = 'DATASET'
    _read_option(config, dataset_section, 'target', optional=False)
    _read_option(config, dataset_section, 'train_start', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'train_end', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'test_start', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'test_end', optional=False, value_type=TYPE_DATE)
    _read_option(config, dataset_section, 'features', optional=False, value_type=TYPE_LIST)

    dataset_section = "ML"
    _read_option(config, dataset_section, 'model', optional=False)


TYPE_STRING = 1
TYPE_INT = 2
TYPE_FLOAT = 3
TYPE_BOOLEAN = 4
TYPE_DATE = 5
TYPE_LIST = 6


def _read_option(config, section, option, value_type=TYPE_STRING, target=None, optional=True, date_format='%Y.%m.%d'):
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
        if value_type == TYPE_STRING:
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
        elif value_type == TYPE_LIST:
            value = config.get(section, option)
            value = [item.strip() for item in value.split(',')]
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
