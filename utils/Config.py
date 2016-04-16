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
dataset_learn_start = None
dataset_learn_end = None
dataset_test_start = None
dataset_test_end = None


def read_config(config_file):
    """ Reads the config file and initializes the config variables.

    Args:
        config_file (str): The relative filepath to the config file.
    """

    config = ConfigParser()
    config.read(config_file)

    database_section = 'DATABASE'
    if config.has_section(database_section):
        _read_option(config, database_section, 'dialect', optional=False)
        _read_option(config, database_section, 'name', optional=False)
        _read_option(config, database_section, 'user')
        _read_option(config, database_section, 'user_password')
        _read_option(config, database_section, 'host')
        _read_option(config, database_section, 'port')
    else:
        raise ConfigError(
            '%s Section not found in Config file. Database information must be provided.' % database_section)

    logging_section = 'LOGGING'
    if config.has_section(logging_section):
        _read_option(config, logging_section, 'level')
        _read_option(config, logging_section, 'file')
        _read_option(config, logging_section, 'override', value_type=TYPE_BOOLEAN)
        _read_option(config, logging_section, 'format')
        _read_option(config, logging_section, 'date_format')

    repository_section = 'REPOSITORY'
    if config.has_section(repository_section):
        _read_option(config, repository_section, 'name', optional=False)
    else:
        raise ConfigError(
            '%s Section not found in Config file. Repository information must be provided.' % repository_section)

    dataset_section = 'DATASET'
    if config.has_section(dataset_section):
        _read_option(config, dataset_section, 'learn_start', value_type=TYPE_DATE)
        _read_option(config, dataset_section, 'learn_end', value_type=TYPE_DATE)
        _read_option(config, dataset_section, 'test_start', value_type=TYPE_DATE)
        _read_option(config, dataset_section, 'test_end', value_type=TYPE_DATE)

TYPE_STRING = 1
TYPE_INT = 2
TYPE_FLOAT = 3
TYPE_BOOLEAN = 4
TYPE_DATE = 5


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
        raise ConfigError("Obligatory option %s not found in section %s" % (option, section))


class ConfigError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
