import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel


class TestModel(unittest.TestCase):
    """
    Unit tests for the DelayModel class.

    Attributes
    ----------
    FEATURES_COLS : list of str
        Expected feature columns after preprocessing.
    TARGET_COL : list of str
        Expected target column after preprocessing.
    """

    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]

    def setUp(self) -> None:
        """
        Initialize the DelayModel instance and load the dataset before each test.

        Returns
        -------
        None
        """
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="data/data.csv")

    def test_model_preprocess_for_training(self):
        """
        Test the preprocess method when preparing features and target for training.

        Checks that the output features and target are pandas DataFrames with expected
        columns and shapes.

        Returns
        -------
        None
        """
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(self):
        """
        Test the preprocess method when preparing features only for serving/inference.

        Checks that the output features are a pandas DataFrame with expected columns.

        Returns
        -------
        None
        """
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(self):
        """
        Test the fit method by training the model and validating performance.

        Steps:
        - Preprocess data to obtain features and target.
        - Split into training and validation sets.
        - Fit the model on training data.
        - Predict on validation data.
        - Generate classification report.
        - Assert recall and f1-score thresholds on classes 0 and 1.

        Returns
        -------
        None
        """
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        features_train, features_validation, target_train, target_validation = train_test_split(
            features,
            target,
            test_size=0.33,
            random_state=42,
        )

        self.model.fit(
            features=features_train,
            target=target_train
        )

        predicted_target = self.model._model.predict(features_validation)

        report = classification_report(target_validation, predicted_target, output_dict=True)

        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(self):
        """
        Test the predict method to ensure output is a list of integer predictions.

        Steps:
        - Preprocess the entire dataset.
        - Fit the model on the full dataset.
        - Predict using the fitted model.
        - Assert output type and length.

        Returns
        -------
        None
        """
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        self.model.fit(
            features=features,
            target=target,
        )

        predicted_targets = self.model.predict(features=features)

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)
