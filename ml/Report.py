#!/usr/bin/python
# coding=utf-8
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, median_absolute_error, \
    r2_score


# TODO: Verteilung der FEhler
# TODO: Beispiele Versionen (als liste übergeben oder so)
# TODO: Score im Verhältnis zu z.B. Alter der Files setzen oder zum Alter des Repos

class Report:
    def __init__(self, ground_truth, predicted):
        self.ground_truth = ground_truth
        self.predicted = predicted
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
        output = ""
        output += "Predicted:\n" + str(self.predicted)
        output += "\nground_truth:" + str(self.ground_truth)
        output += "\nABS DIFF:" + str(abs(self.predicted - self.ground_truth))
        if self.evs is not None:
            output += "\nExplained variance score:\t%f\t(Best is 1.0, lower is worse)" % self.evs
        if self.mse is not None:
            output += "\nMean squared error:\t\t\t%f\t(Best is 0.0, higher is worse)" % self.mse
        if self.mae is not None:
            output += "\nMean absolute error:\t\t%f\t(Best is 0.0, higher is worse)" % self.mae
        if self.mde is not None:
            output += "\nMedian absolute error:\t\t%f\t(Best is 0.0, higher is worse)" % self.mde
        if self.r2s is not None:
            output += "\nR2 Score:\t\t\t\t\t%f\t(Best is 1.0, lower is worse)" % self.r2s
        return output


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


class Report:
    def __init__(self, ground_truth, predicted):
        self.ground_truth = ground_truth
        self.predicted = predicted
        self.evs = None
        self.mse = None
        self.mae = None
        self.mde = None
        self.r2s = None

        self.update()

    def update(self):
        self.evs = explained_variance_score(self.ground_truth, self.predicted)
        self.mse = mean_squared_error(self.ground_truth, self.predicted)
        self.mae = mean_absolute_error(self.ground_truth, self.predicted)
        self.mde = median_absolute_error(self.ground_truth, self.predicted)
        self.r2s = r2_score(self.ground_truth, self.predicted)

    # def test_version(self, version):

    def __str__(self):
        output = ""
        output += "Predicted:\n" + str(self.predicted)
        output += "\nground_truth:" + str(self.ground_truth)
        output += "\nABS DIFF:" + str(abs(self.predicted - self.ground_truth))
        if self.evs is not None:
            output += "\nExplained variance score:\t%f\t(Best is 1.0, lower is worse)" % self.evs
        if self.mse is not None:
            output += "\nMean squared error:\t\t\t%f\t(Best is 0.0, higher is worse)" % self.mse
        if self.mae is not None:
            output += "\nMean absolute error:\t\t%f\t(Best is 0.0, higher is worse)" % self.mae
        if self.mde is not None:
            output += "\nMedian absolute error:\t\t%f\t(Best is 0.0, higher is worse)" % self.mde
        if self.r2s is not None:
            output += "\nR2 Score:\t\t\t\t\t%f\t(Best is 1.0, lower is worse)" % self.r2s
        return output
