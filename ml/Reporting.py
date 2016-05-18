#!/usr/bin/python
# coding=utf-8
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from terminaltables import AsciiTable as Table

# TODO: Verteilung der FEhler
# TODO: Beispiele Versionen (als liste übergeben oder so)
# TODO: Score im Verhältnis zu z.B. Alter der Files setzen oder zum Alter des Repos
from ml import Model
from utils import Config

SCORE_EVS = "evs"  # Explained Variance Score
SCORE_MSE = "mse"  # Mean Equared Error
SCORE_MAE = "mae"  # Mean Absolute Error
SCORE_MDE = "mde"  # MeDian absolute Error
SCORE_R2S = "r2s"  # R^2 Score


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
            output_data.append([_format_float(self.evs), "Explained variance score", "Best is 1.0, lower is worse"])
        if self.mse is not None:
            output_data.append([_format_float(self.mse), "Mean squared error", "Best is 0.0, higher is worse"])
        if self.mae is not None:
            output_data.append([_format_float(self.mae), "Mean absolute error", "Best is 0.0, higher is worse"])
        if self.mde is not None:
            output_data.append([_format_float(self.mde), "Median absolute error", "Best is 0.0, higher is worse"])
        if self.r2s is not None:
            output_data.append([_format_float(self.r2s), "R2 Score", "Best is 1.0, lower is worse"])
        table = Table(output_data)
        table.title = "Report"
        if self.label:
            table.title += ": " + self.label
        return table.table


def _score_attr_to_string(score_attr):
    if score_attr == SCORE_EVS:
        return "Explained variance score"
    elif score_attr == SCORE_MSE:
        return "Mean squared error"
    elif score_attr == SCORE_MAE:
        return "Mean absolute error"
    elif score_attr == SCORE_MDE:
        return "Median absolute error"
    elif score_attr == SCORE_R2S:
        return "R2 Score"


def _format_float(float_value):
    return "% .4f" % float_value


def get_metrics(ground_truth, predicted):
    """ Calculate all metrics at once.

    Args:
        ground_truth (ndarray): The ground truth target array.
        predicted (ndarray): The predicted target array.

    Returns:
        tuple: The different linear regression metrics.
    """
    return (
        get_explained_variance_score(ground_truth, predicted),
        get_mean_squared_error(ground_truth, predicted),
        get_mean_absolute_error(ground_truth, predicted),
        get_median_absolute_error(ground_truth, predicted),
        get_r2_score(ground_truth, predicted),
    )


def get_explained_variance_score(ground_truth, predicted):
    """ Calculates the explained variance.

    The explained variance is a measure of how well the model can explain the variance of the ground truth.
    The best possible result is 1, lower ist worse.

    Args:
        ground_truth (ndarray): The ground truth target array.
        predicted (ndarray): The predicted target array.

    Returns:
        float: The calculated value.
    """
    return explained_variance_score(ground_truth, predicted)


def get_mean_squared_error(ground_truth, predicted):
    """ Calculates the mean squared error.

    This score represents the mean of the squared residuals between the ground truth and predicted values.
    In short: MSE = mean((ground-truth - predicted)^2)
    The best possible result is 0, higher is worse.

    Args:
        ground_truth (ndarray): The ground truth target array.
        predicted (ndarray): The predicted target array.

    Returns:
        float: The calculated value.
    """
    return mean_squared_error(ground_truth, predicted)


def get_mean_absolute_error(ground_truth, predicted):
    """ Calculates the mean absolute error

    This score represents the mean of the absolute residuals between ground truth and predicted values.
    In short: MAE = mean(|ground-truth - predicted|)
    The best possible result is 0, higher is worse.

    Args:
        ground_truth (ndarray): The ground truth target array.
        predicted (ndarray): The predicted target array.

    Returns:
        float: The calculated value.
    """
    return mean_absolute_error(ground_truth, predicted)


def get_median_absolute_error(ground_truth, predicted):
    """ Calculates the median absolute error

    This score represents the median of the absolute residuals between ground truth and predicted values.
    In short: MAE = median(|ground-truth[0] - predicted[0]|, ..., |ground-truth[n] - predicted[n]|)
    The best possible result is 0, higher is worse.

    Args:
        ground_truth (ndarray): The ground truth target array.
        predicted (ndarray): The predicted target array.

    Returns:
        float: The calculated value.
    """
    return median_absolute_error(ground_truth, predicted)


def get_r2_score(ground_truth, predicted):
    """ Calculates the R^2 score (a.k.a. Coefficient of determination)

    This score represents the quality of the prediction by how much it explains the ground truths variance.
    The best possible result is 1, lower is worse.

    Args:
        ground_truth (ndarray): The ground truth target array.
        predicted (ndarray): The predicted target array.

    Returns:
        float: The calculated value.
    """
    return r2_score(ground_truth, predicted)


def get_report_comparisation_table(reports, score_attrs=SCORE_R2S):
    """ Returns a formatted table which compares a list of reports by one or more attributes.

    Args:
        reports (list[Report]): The reports to compare
        score_attrs (list[str]|str): The attribute (or list of attributes) to compare. Use the SCORE_X constants.

    Returns:
        (Table): A table with the data.
    """
    if type(score_attrs) != list:
        score_attrs = [score_attrs]
    multiple_attrs = len(score_attrs) > 1

    headers = []
    if multiple_attrs:
        headers += [""]
    headers += [report.label for report in reports]

    compare_table = [headers]
    for score_attr in score_attrs:
        values = []
        if multiple_attrs:
            values.append(score_attr)
        values += [_format_float(getattr(report, score_attr)) for report in reports]
        compare_table.append(values)
    table = Table(compare_table)

    if multiple_attrs:
        table.title = "Comparisation"
    else:
        table.title = _score_attr_to_string(score_attrs[0]) + " comparisation"
    return table


def save_report_file(content, filename="report", timestamp=True, file_ext="txt", directory=None,
                     timestamp_format="%Y_%m_%d_%H_%M", config=True):
    """ Saves a report string to a file.

    Args:
        content (str): The report data as a string.
        filename (str): The filename to use (without extension)
        timestamp (bool): If a timestamp should be appended to the filename.
        file_ext (str): The file extension to use.
        directory (str): The directory path to save the file in. If None, the working dir will be used.
        timestamp_format (str): The strftime format for the timestamp, which will be appended to the filename.
    """
    if not directory:
        directory = os.getcwd()
    if timestamp:
        time_str = datetime.now().strftime(timestamp_format)
        filename += "_" + time_str
    filepath = os.path.join(directory, filename + "." + file_ext)

    with open(filepath, 'w') as f:
        f.write(content)


def get_top_features_table(model, features, n):
    """ Returns a formatted table which lists the n most-weighted feature of a model.

    Note that this is not applicable with all model types. SVR models for example, doesn't offer the coef_ attribute, as
    they use a kernel trick.
    Also, if polynomial features where used, the coef_s, while avaiable, will not be able to be mapped to the features
    from the feature list.

    Args:
        model: A learned model.
        features (list[str]): A list of feature IDs.
        n (int): How many features should be displayed.

    Returns:
        (Table): A table with the data.
    """
    logging.debug("Calculating top features.")
    for step in model.steps:
        _model = step[1]
        logging.debug("Trying to get top features from step " + step[0])
        if hasattr(_model, 'coef_'):
            try:
                logging.debug("Step %s has %i coefficients." % (step[0], len(_model.coef_)))
                sorted_enum = sorted(enumerate(_model.coef_), key=lambda x: abs(x[1]), reverse=True)
                n = min(n, len(sorted_enum))

                table_data = [["Coefficient", "Feature"]]
                for idx, coef in sorted_enum[:n]:
                    table_data.append([_format_float(coef), features[idx]])
                table = Table(table_data)
                table.title = "Top weighted features"
                return table
            except:
                pass
    return None


def get_category_table(ground_truth, predicted, categories=None, label=None):
    if categories is None:
        categories = [0, 1, 2, 4]
    hits = {}
    misses = {}
    for value_pair in zip(ground_truth, predicted):
        true_category = get_category(value_pair[0], categories)
        predicted_category = get_category(value_pair[1], categories)

        if true_category == predicted_category:
            hits[true_category] = hits.get(true_category, 0) + 1
        else:
            misses[true_category] = misses.get(true_category, 0) + 1

    table_data = [["Category", "Hits", "Misses", "Total"]]
    for i, category in enumerate(categories):
        hit_count = hits.get(category, 0)
        miss_count = misses.get(category, 0)

        cat_str = str(category)
        if i == len(categories) - 1:
            cat_str += "+"
        elif categories[i + 1] - category > 1:
            cat_str += "-" + str(categories[i + 1] - 1)

        record = [cat_str, str(hit_count), str(miss_count), str(hit_count + miss_count)]
        table_data.append(record)
    total_hits = sum(hits.values())
    total_misses = sum(misses.values())
    table_data.append(["Total", str(total_hits), str(total_misses), str(total_hits + total_misses)])
    table = Table(table_data)
    table.title = "Categoric rating"
    if label:
        table.title += ": " + label
    table.inner_footing_row_border = True
    return table


def get_category(value, categories):
    value = max(round(value), 0)
    for i, category in enumerate(categories[:-1]):
        if value < categories[i + 1]:
            return category
    return categories[-1]


def get_config_table():
    config_attrs = [attr for attr in dir(Config) if
                    not callable(getattr(Config, attr)) and  # No functions please
                    not attr.startswith("_") and             # No internal/private attributes
                    not attr.isupper() and                   # No constants
                    not attr.startswith('database') and      # Database config is not so relevant
                    not attr.startswith('reporting')         # Reporting config is not so relevant
                    ]

    table_data = [["Attribute", "Value"]] + [[attr, str(getattr(Config, attr))] for attr in config_attrs]
    table = Table(table_data)
    table.title = "Configuration"
    return table


def plot_target_histogram(dataset, save=False, display=True, filename='target_histogram'):
    """ Plots a histogram of the datasets target vector and also displays a logarithmic curve of it.

    Args:
        dataset(ml.Dataset.Dataset): The dataset from which to plot the taret.
        save (bool): If the plot should be saved.
        display (bool): If the plot should be displayed
        filename (str): The filename to be used when the plot is saved. Without extension.
    """
    logging.info("Plotting target histogram for dataset %s" % dataset.label)
    histogram = {}

    for value in dataset.target.ravel():
        histogram[value] = histogram.get(value, 0) + 1

    x = np.arange(len(histogram.keys()))
    y = [histogram[key] for key in sorted(histogram.keys())]
    y_log = np.log(y)

    fig, ax1 = plt.subplots()

    # Histogram bar plot
    ax1.bar(x, y, align='center', width=0.8, label='Target data')
    ax1.set_xlabel("Bugs per " + dataset.target_id.lower())
    ax1.set_ylabel("Sample Count")
    ax1.legend(loc=1)

    # Plot Log curve
    ax2 = ax1.twinx()
    ax2.bar(x, y_log, align='center', width=0.4, color='red', label='Logarithmic transformation')
    ax2.legend(loc=4)

    title = 'Target Histogram'
    if dataset.label:
        title += " (" + dataset.label + " Set)"
    plt.title(title)
    plt.grid(True)

    if display:
        plt.show()
    if save:
        plt.savefig(filename + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".png", dpi=400)
    plt.clf()


def plot_validation_curve(model_type, train_dataset, score_attr=None, cv=None, alpha_range=None, C_range=None,
                          kernel=None, n_jobs=-1, save=False, display=True, filename="validation_curve"):
    if not save and not display:
        return

    model_type = model_type.upper()
    if model_type == Model.MODEL_TYPE_RIDREG:
        estimator = Model.create_ridge_model()
        param_name = "alpha"
        param_range = alpha_range
    elif model_type == Model.MODEL_TYPE_SVR:
        estimator = Model.create_svr_model(kernel=kernel)
        param_name = "C"
        param_range = C_range
    else:
        logging.warning("Validation curve is not applicable to Model type %s." % model_type)
        return

    logging.info("Calculating validation curve")
    train_scores, valid_scores = validation_curve(
        estimator=estimator,
        X=train_dataset.data,
        y=train_dataset.target,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=score_attr,
        n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    logging.debug("Displaying validation curve")
    plt.title("Validation curve")
    plt.xlabel(param_name)
    plt.ylabel(score_attr.upper() if score_attr else "" + "Score")
    plt.semilogx(param_range, train_scores_mean, 'o-', label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.semilogx(param_range, valid_scores_mean, 'o-', label="Cross-Validation score", color="g")
    plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.2,
                     color="g")
    plt.legend(loc="best")

    if display:
        plt.show()
    if save:
        plt.savefig(filename + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".png", dpi=400)
    plt.clf()


def plot_learning_curve(model_type, train_dataset, train_sizes=np.linspace(.1, 1.0, 5), score_attr=None,
                        cross_validation=False, cv=None, alpha=None, alpha_range=None, C=None, C_range=None,
                        kernel=None, n_jobs=-1, save=False, display=True, filename="learning_curve"):
    if not save and not display:
        return

    estimator = Model.create_model(
        model_type=model_type,
        cross_validation=cross_validation,
        alpha=alpha,
        alpha_range=alpha_range,
        C=C,
        C_range=C_range,
        kernel=kernel,
    )

    logging.info("Calculating learning curve")
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=estimator,
        X=train_dataset.data,
        y=train_dataset.target,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=n_jobs,
        scoring=score_attr,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    logging.debug("Displaying learning curve")
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel(score_attr.upper() if score_attr else "" + "Score")
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score", color="r")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.plot(train_sizes, valid_scores_mean, 'o-', label="Cross-Validation score", color="g")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.2,
                     color="g")
    plt.legend(loc="best")

    if display:
        plt.show()
    if save:
        plt.savefig(filename + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".png", dpi=400)
    plt.clf()
