import unittest

import test_datasets
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing.data import StandardScaler

from ml import Model, Predict, Reporting


class ModelTestCase(unittest.TestCase):
    def _test_dataset(self, model, train_dataset, test_dataset):
        model.fit(train_dataset.data, train_dataset.target)
        train_prediction = Predict.predict_with_model(train_dataset, model)
        test_prediction = Predict.predict_with_model(test_dataset, model)

        train_report = Reporting.Report(train_dataset.target, train_prediction, "Training")
        test_report = Reporting.Report(test_dataset.target, test_prediction, "Test")
        comparisation_table = Reporting.get_report_comparisation_table(
            [train_report, test_report],
            [Reporting.SCORE_R2S, Reporting.SCORE_MAE, Reporting.SCORE_MDE, Reporting.SCORE_EVS])
        print(comparisation_table.table)

        self.assertAlmostEqual(train_report.r2s, 1.0)
        self.assertAlmostEqual(train_report.mae, 0.0)
        self.assertAlmostEqual(test_report.r2s, 1.0)
        self.assertAlmostEqual(test_report.mae, 0.0)


class TestLinearRegression(ModelTestCase):
    def test_simple_linear_dataset(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_LINREG
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_normalized(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_LINREG,
            feature_scaling=True
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_poly(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_LINREG,
            polynomial_degree=2
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_full_pipeline(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_LINREG,
            feature_scaling=True,
            polynomial_degree=2
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)



class TestRidgeRegression(ModelTestCase):
    def test_simple_linear_dataset(self):
        pass

    def test_simple_linear_dataset_normalized(self):
        pass

    def test_simple_linear_dataset_cross_validation(self):
        pass


class TestSVR(ModelTestCase):
    def test_simple_linear_dataset(self):
        pass

    def test_simple_linear_dataset_normalized(self):
        pass

    def test_simple_linear_dataset_cross_validation(self):
        pass


if __name__ == '__main__':
    unittest.main()
