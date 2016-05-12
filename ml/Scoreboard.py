#!/usr/bin/python
# coding=utf-8
import os
from operator import attrgetter

from utils import Config

# TODO: maybe make this configurable? Low prio.
SCOREBOARD_FILE = 'scores.scoreboard'
SEPARATOR = ";"
FEATURE_SEPARATOR = ","

entries = set()


class ScoreboardEntry:
    def __init__(self, label, evs, mse, mae, mde, r2s, repository_name, ml_model, ml_feature_scaling,
                 ml_polynomial_degree, dataset_use_ngrams, dataset_target, dataset_train_start, dataset_train_end,
                 dataset_test_start, dataset_test_end,
                 dataset_features):
        self.label = label
        self.evs = float(evs)
        self.mse = float(mse)
        self.mae = float(mae)
        self.mde = float(mde)
        self.r2s = float(r2s)
        self.repository_name = repository_name
        self.ml_model = ml_model
        self.ml_feature_scaling = ml_feature_scaling
        self.ml_polynomial_degree = ml_polynomial_degree
        self.dataset_use_ngrams = dataset_use_ngrams
        self.dataset_target = dataset_target
        self.dataset_train_start = dataset_train_start
        self.dataset_train_end = dataset_train_end
        self.dataset_test_start = dataset_test_start
        self.dataset_test_end = dataset_test_end
        self.dataset_features = dataset_features

    def __hash__(self):
        fields = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
        fields = filter(lambda attr: attr not in ('evs', 'mse', 'mae', 'mde', 'r2s'), fields)
        field_values = tuple(str(getattr(self, field)) for field in fields)
        return hash(field_values)

    def __eq__(self, other):
        return hash(self) == hash(other)


def create_entry_from_config(report):
    """ Creates a new ScoreboardEntry Object from a report and the current Config state.

    Args:
        report (Report): The report to get the label and ratings from.

    Returns:
        ScoreboardEntry: A new ScoreboardEntry.
    """
    return ScoreboardEntry(
        report.label,
        report.evs,
        report.mse,
        report.mae,
        report.mde,
        report.r2s,
        Config.repository_name,
        Config.ml_model,
        Config.ml_feature_scaling,
        Config.ml_polynomial_degree,
        Config.dataset_use_ngrams,
        Config.dataset_target,
        Config.dataset_train_start,
        Config.dataset_train_end,
        Config.dataset_test_start,
        Config.dataset_test_end,
        Config.dataset_features,
    )


def parse_entry_from_string(string):
    """ Parses a string into a ScoreboardEntry.

    Args:
        string (str): The string representing a storeboard entry, the data separated by SEPARATOR.

    Returns:
        ScoreboardEntry: A new ScoreboardEntry.
    """
    # find out how many arguments constructor needs, discard the rest
    argcount = ScoreboardEntry.__init__.__code__.co_argcount
    args = [fragment.strip() for fragment in string.split(SEPARATOR)]
    if len(args) > argcount:
        args = args[:argcount]
    args[-1] = args[-1].split(FEATURE_SEPARATOR)
    return ScoreboardEntry(*args)


def parse_entry_to_string(scoreboard_entry):
    """ Parses a ScoreboardEntry into a string representation of its data.

    Args:
        scoreboard_entry (ScoreboardEntry): The ScoreboardEntry object to parse.

    Returns:
        str: The string representation of the ScoreboardEntries data.
    """
    return SEPARATOR.join([
        str(scoreboard_entry.label),
        str(scoreboard_entry.evs),
        str(scoreboard_entry.mse),
        str(scoreboard_entry.mae),
        str(scoreboard_entry.mde),
        str(scoreboard_entry.r2s),
        str(scoreboard_entry.repository_name),
        str(scoreboard_entry.ml_model),
        str(scoreboard_entry.ml_feature_scaling),
        str(scoreboard_entry.ml_polynomial_degree),
        str(scoreboard_entry.dataset_use_ngrams),
        str(scoreboard_entry.dataset_target),
        str(scoreboard_entry.dataset_train_start),
        str(scoreboard_entry.dataset_train_end),
        str(scoreboard_entry.dataset_test_start),
        str(scoreboard_entry.dataset_test_end),
        FEATURE_SEPARATOR.join(scoreboard_entry.dataset_features)])


def read_entries():
    """ Reads a list of scoreboard entries from the scoreboard file into the global variable entries """
    if os.path.isfile(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, mode='r') as f:
            for line in f:
                e = parse_entry_from_string(line)
                entries.add(e)


def add_entry(entry):
    """ Add a new entry to the scoreboard.

    Note that it won't be written to the scoreboard file before calling write_entries()

    Args:
        entry (ScoreboardEntry): The Scoreboard Entry to write.
    """
    global entries
    if not entries:
        read_entries()
    entries.add(entry)


def write_entries():
    """ Persist the current scoreboard entries into the scoreboard file """
    global entries
    with open(SCOREBOARD_FILE, mode='w') as f:
        f.writelines([parse_entry_to_string(entry) + "\n" for entry in entries])


# Rating Attribute structure: (attribute_name, reverse_ordering)
RATING_ATTRIBUTE_EVS = ("evs", True)
RATING_ATTRIBUTE_MSE = ("mse", False)
RATING_ATTRIBUTE_MAE = ("mae", False)
RATING_ATTRIBUTE_MDE = ("mde", False)
RATING_ATTRIBUTE_R2S = ("r2s", True)


def get_ranking(entry, rating_attribute):
    """ Get the ranking of an entry in the current scoreboard, according to a specific attribute.

    Args:
        entry (ScoreboardEntry): The scoreboard entry to get the ranking of.
        rating_attribute (tuple(str, bool)): The attribute by which to rank the entries.
            Use one of the RATING_ATTRIBUT_X constants.

    Returns:
        int: The rank of the entry.
    """
    global entries
    if not entries:
        read_entries()

    sorted_entries = sorted(
        list(entries),
        key=attrgetter(rating_attribute[0]),
        reverse=rating_attribute[1]
    )
    return sorted_entries.index(entry)
