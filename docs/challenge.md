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


## Part II

## Part III

## Part IV

