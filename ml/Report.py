#!/usr/bin/python
# coding=utf-8
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from terminaltables import AsciiTable as Table



# TODO: Verteilung der FEhler
# TODO: Beispiele Versionen (als liste übergeben oder so)
# TODO: Score im Verhältnis zu z.B. Alter der Files setzen oder zum Alter des Repos

class Report:
    def __init__(self, ground_truth, predicted, label=""):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.label = label
        self.evs = None
        self.mse = None
        self.mae = None
        self.mde = None
        self.r2s = None

        self.update()

    def update(self):
        self.evs = get_explained_variance_score(self.ground_truth, self.predicted)
        self.mse = get_mean_squared_error(self.ground_truth, self.predicted)
        self.mae = get_mean_absolute_error(self.ground_truth, self.predicted)
        self.mde = get_median_absolute_error(self.ground_truth, self.predicted)
        self.r2s = get_r2_score(self.ground_truth, self.predicted)

    def __str__(self):
        output_data = [
            ["Value", "Description", "Info"]
        ]
        if self.evs is not None:
            output_data.append([str(self.evs), "Explained variance score", "Best is 1.0, lower is worse"])
        if self.mse is not None:
            output_data.append([str(self.mse), "Mean squared error", "Best is 0.0, higher is worse"])
        if self.mae is not None:
            output_data.append([str(self.mae), "Mean absolute error", "Best is 0.0, higher is worse"])
        if self.mde is not None:
            output_data.append([str(self.mde), "Median absolute error", "Best is 0.0, higher is worse"])
        if self.r2s is not None:
            output_data.append([str(self.r2s), "R2 Score", "Best is 1.0, lower is worse"])
        table = Table(output_data)
        table.title = "Report"
        if self.label:
            table.title += ": " + self.label
        return table.table


def get_metrics(ground_truth, predicted):
    return (
        get_explained_variance_score(ground_truth, predicted),
        get_mean_squared_error(ground_truth, predicted),
        get_mean_absolute_error(ground_truth, predicted),
        get_median_absolute_error(ground_truth, predicted),
        get_r2_score(ground_truth, predicted),
    )


def get_explained_variance_score(ground_truth, predicted):
    return explained_variance_score(ground_truth, predicted)


def get_mean_squared_error(ground_truth, predicted):
    return mean_squared_error(ground_truth, predicted)


def get_mean_absolute_error(ground_truth, predicted):
    return mean_absolute_error(ground_truth, predicted)


def get_median_absolute_error(ground_truth, predicted):
    return median_absolute_error(ground_truth, predicted)


def get_r2_score(ground_truth, predicted):
    return r2_score(ground_truth, predicted)