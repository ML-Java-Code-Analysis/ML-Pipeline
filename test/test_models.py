import unittest

import test_datasets
import numpy as np

from ml import Model, Predict, Reporting


class ModelTestCase(unittest.TestCase):
    def _test_dataset(self, model, train_dataset, test_dataset, precision=7):
        model.fit(train_dataset.data, train_dataset.target)
        train_prediction = Predict.predict_with_model(train_dataset, model)
        test_prediction = Predict.predict_with_model(test_dataset, model)

        train_report = Reporting.Report(train_dataset.target, train_prediction, "Training")
        test_report = Reporting.Report(test_dataset.target, test_prediction, "Test")
        comparisation_table = Reporting.get_report_comparisation_table(
            [train_report, test_report],
            [Reporting.SCORE_R2S, Reporting.SCORE_MAE, Reporting.SCORE_MDE, Reporting.SCORE_EVS])
        print(comparisation_table.table)

        self.assertAlmostEqual(train_report.r2s, 1.0, places=precision)
        self.assertAlmostEqual(train_report.mae, 0.0, places=precision)
        self.assertAlmostEqual(test_report.r2s, 1.0, places=precision)
        self.assertAlmostEqual(test_report.mae, 0.0, places=precision)


class TestLinearRegression(ModelTestCase):
    def test_simple_linear_dataset(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_LINREG
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_scaled(self):
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
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            alpha=0
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_scaled(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            feature_scaling=True,
            alpha=0
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_poly(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            polynomial_degree=2,
            alpha=0
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_full_pipeline(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            polynomial_degree=2,
            feature_scaling=True,
            alpha=0.5
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset, precision=0)


class TestRidgeRegressionCV(ModelTestCase):
    def test_simple_linear_dataset(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            alpha_range=[0, 0.01, 0.1, 1, 10, 100, 1000]
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset, 2)

    def test_simple_linear_dataset_full_pipeline(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            feature_scaling=False,
            polynomial_degree=1,
            alpha_range=[0, 0.01, 0.1, 1, 10, 100, 1000]
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset, 0)


class TestSVR(ModelTestCase):
    def test_simple_linear_dataset(self):
        pass

    def test_simple_linear_dataset_scaled(self):
        pass

    def test_simple_linear_dataset_cross_validation(self):
        pass


if __name__ == '__main__':
    unittest.main()
