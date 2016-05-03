#!/usr/bin/python
# coding=utf-8
import os

from utils import Config

SCOREBOARD_FILE = 'scores.scoreboard'
SEPARATOR = ";"
FEATURE_SEPARATOR = ","

entries = set()


class ScoreboardEntry:
    def __init__(self, evs, mse, mae, mde, r2s, repository_name, ml_model, ml_normalize, dataset_target,
                 dataset_train_start, dataset_train_end, dataset_test_start, dataset_test_end, dataset_features):
        self.evs = evs
        self.mse = mse
        self.mae = mae
        self.mde = mde
        self.r2s = r2s
        self.repository_name = repository_name
        self.ml_model = ml_model
        self.ml_normalize = ml_normalize
        self.dataset_target = dataset_target
        self.dataset_train_start = dataset_train_start
        self.dataset_train_end = dataset_train_end
        self.dataset_test_start = dataset_test_start
        self.dataset_test_end = dataset_test_end
        self.dataset_features = dataset_features

    def __hash__(self):
        fields = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
        field_values = tuple(str(getattr(self, field)) for field in fields)
        return hash(field_values)

    def __eq__(self, other):
        return hash(self) == hash(other)



def create_entry_from_config(report):
    return ScoreboardEntry(
        report.evs,
        report.mse,
        report.mae,
        report.mde,
        report.r2s,
        Config.repository_name,
        Config.ml_model,
        Config.ml_normalize,
        Config.dataset_target,
        Config.dataset_train_start,
        Config.dataset_train_end,
        Config.dataset_test_start,
        Config.dataset_test_end,
        Config.dataset_features,
    )


def parse_entry_from_string(string):
    # find out how many arguments constructor needs, discard the rest
    argcount = ScoreboardEntry.__init__.__code__.co_argcount
    args = [fragment.strip() for fragment in string.split(SEPARATOR)]
    if len(args) > argcount:
        args = args[:argcount]
    args[-1] = args[-1].split(FEATURE_SEPARATOR)
    return ScoreboardEntry(*args)


def parse_entry_to_string(scoreboard_entry):
    return SEPARATOR.join([
        str(scoreboard_entry.evs),
        str(scoreboard_entry.mse),
        str(scoreboard_entry.mae),
        str(scoreboard_entry.mde),
        str(scoreboard_entry.r2s),
        str(scoreboard_entry.repository_name),
        str(scoreboard_entry.ml_model),
        str(scoreboard_entry.ml_normalize),
        str(scoreboard_entry.dataset_target),
        str(scoreboard_entry.dataset_train_start),
        str(scoreboard_entry.dataset_train_end),
        str(scoreboard_entry.dataset_test_start),
        str(scoreboard_entry.dataset_test_end),
        FEATURE_SEPARATOR.join(scoreboard_entry.dataset_features)])


def add_entry(entry):
    global entries
    if os.path.isfile(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, mode='r') as f:
            for line in f:
                e = parse_entry_from_string(line)
                print(hash(e))
                entries.add(e)
    entries.add(entry)


def write_entries():
    global entries
    with open(SCOREBOARD_FILE, mode='w') as f:
        f.writelines([parse_entry_to_string(entry) + "\n" for entry in entries])
