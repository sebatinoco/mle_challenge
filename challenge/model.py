import pandas as pd
import numpy as np
from typing import Tuple, Union, List
import xgboost as xgb
import pickle

from .utils import get_min_diff

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

        # Columns to be used as features
        self.top_10_features = [
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

        # Target column name
        self._target_name = "delay"

        # Model file path
        self._model_path = 'challenge/models/xgb_model.pkl'

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        # Generate the target column based on the time difference between flights
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data[self._target_name] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        # One-hot encoding for categorical features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

        # Filter to top 10 features
        features = features[self.top_10_features] 

        # Get target column
        target = data[[self._target_name]]

        return (features, target) if target_column else features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Get scale for imbalanced classes
        n_y0 = len(target[target[self._target_name] == 0])
        n_y1 = len(target[target[self._target_name] == 1])
        scale = n_y0/n_y1

        # Fit XGBoost model
        xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        xgb_model.fit(features, target)

        # Export model to pickle file
        self.export_model(xgb_model, self._model_path)

        # Save model in the class attribute
        self._model = xgb_model 

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        if self._model is None:
            self.load_model(self._model_path)

        return self._model.predict(features).tolist()

    def export_model(
        self,
        model: xgb.XGBClassifier,
        model_path: str
    ) -> None:
        """
        Export the trained model to a file.

        Args:
            model (xgb.XGBClassifier): trained XGBoost model.
            model_path (str): path to save the model file.
        
        Returns:
            None
        """

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(
        self,
        model_path: str
    ) -> None:
        """
        Load a pre-trained model from a file.

        Args:
            model_path (str): path to the model file.
        
        Returns:
            None
        """

        try:
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model has been trained and exported correctly.")