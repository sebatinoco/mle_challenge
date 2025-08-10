# MLE Challenge

Welcome! This document provides comprehensive documentation to understand the solution developed for the Machine Learning Engineer challenge.

It is organized into four sections, corresponding to each part of the challenge:

- [Part I](#part-i)  
- [Part II](#part-ii)  
- [Part III](#part-iii)  
- [Part IV](#part-iv)  

Let's dive in!


<p align="center">
<img src="https://media4.giphy.com/media/v1.Y2lkPTZjMDliOTUyMWc5YW94d2FlM3ZlMncwMTVueHdiNHIxOWdzM2FheGhrNmQ4cnhzZyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3aGZA6WLI9Jde/source.gif" alt="Lets gooo" width="400" />
</p>


## Part I

[Back to the start](#mle-challenge)

To complete this section, several fixes were required to ensure smooth execution and correctness. Below is a summary of the key changes made across the provided files:

- **exploration.ipynb:**  
    - Fixed incorrect `sns.barplot` usage by specifying keyword arguments explicitly:  
    ```python
    # Original call causing execution error
    sns.barplot(flights_by_airline.index, flights_by_airline.values)  

    # Fixed call with keyword arguments
    sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values)  
    ```

- **requirements.txt:**  
    - Updated `numpy` version from `1.22.4` to `1.25` to prevent compatibility errors.  
    - Added `xgboost==3.0.3` to support model training.

- **test_model.py:**  
    - *setUp*: Corrected the test data path from `"../data/data.csv"` to `"data/data.csv"`.  
    - *test_model_fit*: Adjusted training procedure to use only the training split, avoiding data leakage and biased metrics:  
    ```python
    # Original code (uses full dataset for training)
    _, features_validation, _, target_validation = train_test_split(features, target, test_size=0.33, random_state=42)
    self.model.fit(
        features=features,
        target=target
    )

    # Fixed code (trains only on training subset)
    features_train, features_validation, target_train, target_validation = train_test_split(
        features, target, test_size=0.33, random_state=42
    )
    self.model.fit(
        features=features_train,
        target=target_train
    )
    ```

---

With these fixes applied, the main remaining task is to complete the implementation of the **model.py** script. Below is a summary highlighting the work done in this module:

1. **Complete Method Implementations**  
   - **`preprocess`**:  
     - Generates a binary target column `delay` based on a 15-minute time difference threshold using the external helper `get_min_diff`.  
     - Performs one-hot encoding on categorical variables (`OPERA`, `TIPOVUELO`, `MES`).  
     - Selects a predefined set of top 10 features for model input.  
     - Returns features alone or both features and target, depending on the `target_column` parameter.

   - **`fit`**:  
     - Trains an XGBoost classifier with automatic class imbalance adjustment via `scale_pos_weight`.  
     - Saves the trained model to disk using pickle serialization.  
     - Stores the trained model in an internal class attribute.

   - **`predict`**:  
     - Loads the model from disk if it hasn’t been loaded already.  
     - Predicts target labels on new features and returns them as a list.

2. **Model Persistence**  
   - Added explicit methods to **export** (`export_model`) and **load** (`load_model`) the model using pickle.  
   - Includes error handling for cases where the model file is missing.

3. **Feature Engineering and Target Definition**  
   - Implements domain-specific feature engineering by computing delay targets from flight time differences.  
   - Uses one-hot encoding to convert categorical inputs into numeric features suitable for modeling.

4. **Class Attribute Configuration**  
   - Defines feature columns, target name, and model file path as class-level attributes to facilitate easy maintenance and updates.

5. **Enhanced Documentation and Type Hinting**  
   - Provides comprehensive docstrings detailing arguments, return types, and method functionality.  
   - Applies Python type hints consistently across all method signatures.


### Hotfixes

Through the development of the project, some hotfixes were needed to keep the progress of the challenge. Here's a list of the implemented hotfixes related to the scripts of Part I.

#### Hotfix: Adding condition to generate a target vector

As stated in the title, this hotfix aimed to add a condition to generate the vector column. This way, we avoid unnecesary processing (and potential errors) on production.

```python
### PREVIOUS CODE
# Generate the target column based on the time difference between flights
data['min_diff'] = data.apply(get_min_diff, axis = 1)
threshold_in_minutes = 15
data[self._target_name] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

### NEW CODE
# If target_column is provided, generate the target vector
if target_column:
    # Generate the target column based on the time difference between flights
    data['min_diff'] = data.apply(get_min_diff, axis = 1)
    threshold_in_minutes = 15
    data[self._target_name] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

    # Get target column
    target = data[[self._target_name]]
```

#### Hotfix: Adding OneHotEncoder from scikit-learn

Although this was originally intented to prevent further errors from unknown categories, later this was not necessary as the unit tests required to return an error when this happens (LOL).

Either way, this is the change on the code:

```python
### PREVIOUS CODE
# One-hot encoding for categorical features
features = pd.concat([
    pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
    pd.get_dummies(data['MES'], prefix = 'MES')], 
    axis = 1
)

### NEW CODE
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
```

#### Hotfix: Disabling handle_unknown

As mentioned, I had to disable the `handle_unknown='ignore'` parameter to return an error where a unknown category shows up.

This is the change on code:

```python
### PREVIOUS CODE
self._encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

### NEW CODE
self._encoder = OneHotEncoder(sparse_output=False)
```

## Part II

[Back to the start](#mle-challenge)

To deploy the model in an `API` using `FastAPI` and pass all the unit tests, it required to generate code for a correct deployment. Here's a list of the key changes made across the provided files:

- **model.py:**
    - [Hotfix 1](#hotfix-adding-condition-to-generate-a-target-vector), [Hotfix 2](#hotfix-adding-onehotencoder-from-scikit-learn) and [Hotfix 3](#hotfix-disabling-handle_unknown) described on the previous section.

- **requirements.txt**
    - Added `anyio==3.4.0` to requirements.txt to avoid deployment errors.

- **utils.py**:
    - Added `Flight` and `FlightsRequest` pydantic model to structure requests received FastAPI endpoint.

- **api.py:**
    - Initiation of `DelayModel` for prediction.
    - Completed `/predict` endpoint considering:
        - Receive a `FlightRequest` request.
        - Transformation to pandas DataFrame of the input.
        - Preprocess the input using `preprocess` method of `DelayModel`.
        - Generate predictions using `predict` method of `DelayModel`.
        - Have a try-except logic to return a 200 response when there are no errors parsing generating the prediction, and return a 400 response when an error occurs (mainly targeting the unknown categories shown on columns).

## Part III

[Back to the start](#mle-challenge)

With the previous scripts fully functional, we are now ready to deploy our API to the cloud. For this, we use **Cloud Run**, a Google Cloud Platform (GCP) service that simplifies the deployment of containerized applications.

You can access the deployed application here:  
[Deployed Application](https://mle-challenge-756665630445.southamerica-west1.run.app)

To enable deployment, we applied the following key changes to the repository:

- **Dockerfile:**  
  - Updated the Python base image to `python:3.10-slim` to reduce the image size.  
  - Configured the Dockerfile to copy application scripts, install dependencies, and launch the API on port 8000.

- **requirements.txt:**  
  - Upgraded `scikit-learn` from version `1.3.0` to `~1.5.0` to prevent deployment errors.

- **requirements-test.txt:**  
  - Added libraries to support the stress testing phase and avoid related errors:  
    - `Jinja2==3.0.3`  
    - `itsdangerous==2.0.1`  
    - `Werkzeug==2.0.3`

- **Makefile:**  
  - Updated the `STRESS_URL` variable to point to the URL of the deployed API.

- **.dockerignore:**  
  - Added a `.dockerignore` file to exclude unnecessary files from the Docker build context, helping to reduce the image size.

## Part IV

