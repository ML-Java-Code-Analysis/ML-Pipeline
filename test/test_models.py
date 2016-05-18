import inspect
import unittest

import test_datasets
import numpy as np
from sklearn.preprocessing.data import StandardScaler

from ml import Model, Predict, Reporting


class ModelTestCase(unittest.TestCase):
    def _test_dataset(self, model, train_dataset, test_dataset, precision=7, title=None):
        model.fit(train_dataset.data, train_dataset.target)
        train_prediction = Predict.predict_with_model(train_dataset, model)
        test_prediction = Predict.predict_with_model(test_dataset, model)

        train_report = Reporting.Report(train_dataset.target, train_prediction, "Training")
        test_report = Reporting.Report(test_dataset.target, test_prediction, "Test")
        comparisation_table = Reporting.get_report_comparisation_table(
            [train_report, test_report],
            [Reporting.SCORE_R2S, Reporting.SCORE_MAE, Reporting.SCORE_MDE, Reporting.SCORE_EVS])

        if title:
            print("\n" + "=" * 3 + " " + title + " " + "=" * 3)
        print(comparisation_table.table)

        for step in model.steps:
            try:
                print("Coefs: " + str(step[1].coef_))
            except:
                pass

        self.assertAlmostEqual(train_report.r2s, 1.0, places=precision)
        self.assertAlmostEqual(train_report.mae, 0.0, places=precision)
        self.assertAlmostEqual(test_report.r2s, 1.0, places=precision)
        self.assertAlmostEqual(test_report.mae, 0.0, places=precision)


"""
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

    def test_simple_polynomial_dataset_poly(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_LINREG,
            feature_scaling=False,
            polynomial_degree=2,
        )
        train_dataset, test_dataset = test_datasets.get_simple_polynomial_datasets()
        self._test_dataset(model, train_dataset, test_dataset, title=str(type(self)) + "." + inspect.stack()[0][3])


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
            alpha=0
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_linear_dataset_full_pipeline(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            cross_validation=True,
            feature_scaling=False,
            polynomial_degree=1,
            alpha_range=[0, 0.1, 1, 10, 100]
        )

        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()
        self._test_dataset(model, train_dataset, test_dataset)

    def test_simple_polynomial_dataset_full_pipeline(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_RIDREG,
            cross_validation=True,
            feature_scaling=True,
            polynomial_degree=2,
            alpha_range=[0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        )
        train_dataset, test_dataset = test_datasets.get_simple_polynomial_datasets()
        self._test_dataset(model, train_dataset, test_dataset, title=str(type(self)) + "." + inspect.stack()[0][3])
"""


class TestLinearSVR(ModelTestCase):
    def test_simple_linear_dataset(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_SVR,
            cross_validation=False,
            feature_scaling=False,
            C=1,
            kernel=Model.KERNEL_LINEAR
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()

        self._test_dataset(model, train_dataset, test_dataset, 0, title="SVR linear on linear dataset")

    def test_simple_linear_dataset_scaled(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_SVR,
            cross_validation=False,
            feature_scaling=True,
            C=1,
            kernel=Model.KERNEL_LINEAR
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()

        self._test_dataset(model, train_dataset, test_dataset, 0, title="SVR scaled on linear dataset")

    def test_simple_linear_dataset_scaled_cv(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_SVR,
            cross_validation=True,
            feature_scaling=True,
            C_range=[0.001, 0.01, 0.1, 1, 10, 100],
            kernel=Model.KERNEL_LINEAR
        )
        train_dataset, test_dataset = test_datasets.get_simple_linear_datasets()

        self._test_dataset(model, train_dataset, test_dataset, 0, title="SVR scaled CV on linear dataset")

    def test_simple_poly_dataset_scaled_cv(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_SVR,
            cross_validation=True,
            feature_scaling=True,
            C_range=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
            kernel=Model.KERNEL_LINEAR
        )
        train_dataset, test_dataset = test_datasets.get_simple_polynomial_datasets(n=1000)

        self._test_dataset(model, train_dataset, test_dataset, 0, title="SVR scaled CV on poly dataset")


class TestPolynomialSVR(ModelTestCase):
    def test_simple_poly_dataset_scaled_cv(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_SVR,
            cross_validation=True,
            feature_scaling=True,
            C_range=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
            kernel=Model.KERNEL_POLYNOMIAL
        )
        train_dataset, test_dataset = test_datasets.get_simple_polynomial_datasets(n=1000)

        self._test_dataset(model, train_dataset, test_dataset, 0, title="SVR with polynomial kernel, scaled CV on poly dataset")


class TestRbfSVR(ModelTestCase):
    def test_simple_poly_dataset_scaled_cv(self):
        model = Model.create_model(
            model_type=Model.MODEL_TYPE_SVR,
            cross_validation=True,
            feature_scaling=True,
            C_range=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
            kernel=Model.KERNEL_RBF
        )
        train_dataset, test_dataset = test_datasets.get_simple_polynomial_datasets(n=1000)

        scaler = StandardScaler()
        scaler.fit(train_dataset.data)
        print("Train mean: " + str(scaler.transform(train_dataset.data).mean(axis=0)))
        print("Test mean: " + str( scaler.transform(test_dataset.data).mean(axis=0)))
        print("Train std: " + str(scaler.transform(train_dataset.data).std(axis=0)))
        print("Test str: " + str( scaler.transform(test_dataset.data).std(axis=0)))

        self._test_dataset(model, train_dataset, test_dataset, 0, title="SVR with RBF kernel, scaled CV on poly dataset")

if __name__ == '__main__':
    unittest.main()
