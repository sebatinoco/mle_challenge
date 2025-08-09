import pandas as pd
import numpy as np
from typing import Tuple, Union, List
import xgboost as xgb
import pickle
from sklearn.preprocessing import OneHotEncoder

from .utils import get_min_diff

class DelayModel:

    def __init__(
        self
    ):
        self._model: xgb.XGBClassifier = None # Model should be saved in this attribute.
        self._encoder: OneHotEncoder = None # Categorical encoder

        self._feature_names = ["OPERA", "TIPOVUELO", "MES"]

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

        # Models file path
        self._model_path = 'challenge/models/xgb_model.pkl' # XGBoost
        self._encoder_path = 'challenge/models/onehot_encoder.pkl' # Encoder

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

        # If target_column is provided, generate the target vector
        if target_column:
            # Generate the target column based on the time difference between flights
            data['min_diff'] = data.apply(get_min_diff, axis = 1)
            threshold_in_minutes = 15
            data[self._target_name] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

            # Get target column
            target = data[[self._target_name]]

        # Load the encoder if it is not already loaded
        if self._encoder is None:
            try:
                self._encoder = self.load_model(self._encoder_path)
            except FileNotFoundError:
                print("Encoder not found. Fitting a new OneHotEncoder.")
                self._encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self._encoder.fit(data[self._feature_names])
                self.export_model(self._encoder, self._encoder_path) # Save encoder

        # Transform the data using one-hot encoding
        bow = self._encoder.transform(data[self._feature_names])
        features = pd.DataFrame(bow, columns=self._encoder.get_feature_names_out())

        # Filter to top 10 features
        features = features[self.top_10_features] 

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
            self._model = self.load_model(self._model_path)

        return self._model.predict(features).tolist()

    def export_model(
        self,
        model: Union[xgb.XGBClassifier, OneHotEncoder],
        model_path: str
    ) -> None:
        """
        Export the trained model to a file.

        Args:
            model (xgb.XGBClassifier | OneHotEncoder): trained XGBoost model or OneHotEncoder.
            model_path (str): path to save the model file.
        
        Returns:
            None
        """

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_path: str) -> Union[xgb.XGBClassifier, OneHotEncoder]:
            """
            Load model from disk.

            Args:
                model_path: filepath

            Returns:
                Loaded model instance
            """
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Assign to attributes if recognized
            if model_path == self._model_path:
                self._model = model
            elif model_path == self._encoder_path:
                self._encoder = model
            else:
                raise ValueError(f"Unsupported model path: {model_path}")

            return model