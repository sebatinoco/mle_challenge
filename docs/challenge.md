# MLE Challenge

## Useful links

Here's a list of the key components of this project:

- [Deployed API](https://mle-challenge-756665630445.southamerica-west1.run.app)
- [Github repository](https://github.com/sebatinoco/mle_challenge)

## Introduction

Welcome! This document provides comprehensive documentation to understand the solution developed for the Machine Learning Engineer challenge.

It is organized into four sections, corresponding to each part of the challenge:

- [Part I](#part-i)  
- [Part II](#part-ii)  
- [Part III](#part-iii)  
- [Part IV](#part-iv)  

Let's dive in!


<p align="center">
<img src="https://media4.giphy.com/media/v1.Y2lkPTZjMDliOTUyMWc5YW94d2FlM3ZlMncwMTVueHdiNHIxOWdzM2FheGhrNmQ4cnhzZyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3aGZA6WLI9Jde/source.gif" alt="Lets gooo" width="300" />
</p>


## Part I

[Back to the start](#mle-challenge)

To complete this section, several fixes were required to ensure smooth execution and correctness. Below is a summary of the key changes made across the provided files:

<p align="center">
<img src="https://i.pinimg.com/originals/6c/90/28/6c90288d7e10d46d18895f17f420a92c.gif" alt="Working" width="300" />
</p>

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

<p align="center">
<img src="https://media1.tenor.com/m/cxLlTok8ni4AAAAd/hotfix-patch.gif" alt="Working" width="300" />
</p>

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

To deploy the model in an `API` using `FastAPI` and pass all the unit tests, it required to generate code for a correct deployment. 

<p align="center">
<img src="https://media1.tenor.com/m/Mfamt2u-Mb0AAAAC/il-paradiso-delle-signore-vittorio-conti.gif" alt="Working" width="300" />
</p>

Here's a list of the key changes made across the provided files:

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

<p align="center">
<img src="https://media1.tenor.com/m/PulKR2Nm9d4AAAAC/jump-deploying-without-tests.gif" alt="Deployment" width="300" />
</p>

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

Here are the results obtained after running `stress-test`:

| Name        | # reqs | # fails | Avg | Min | Max | Median | req/s | failures/s |
|-------------|--------|---------|-----|-----|-----|--------|-------|------------|
| POST /predict | 3647  | 0 (0.00%) | 488 | 21  | 1018 | 490    | 61.04 | 0.00       |

Aggregated:

| Name       | # reqs | # fails | Avg | Min | Max | Median | req/s | failures/s |
|------------|--------|---------|-----|-----|-----|--------|-------|------------|
| Aggregated | 3647   | 0 (0.00%) | 488 | 21  | 1018 | 490    | 61.04 | 0.00       |

---

Response time percentiles (approximated):

| Type | Name       | 50% | 66% | 75% | 80% | 90% | 95% | 98% | 99% | 99.9% | 99.99% | 100% | # reqs |
|------|------------|-----|-----|-----|-----|-----|-----|-----|-----|-------|--------|------|--------|
| POST | /predict   | 490 | 620 | 720 | 800 | 880 | 920 | 970 | 980 | 1000  | 1000   | 1000 | 3647   |
| None | Aggregated | 490 | 620 | 720 | 800 | 880 | 920 | 970 | 980 | 1000  | 1000   | 1000 | 3647   |

From the results, we conclude:

- The deployed API handled 3,647 requests with zero failures, showing high reliability under load.
- Sadly, the response times were raising :(. This is a sign of potential queuing, probably because of Pandas usage (or maybe lack of resources?).
- Average response time was about 488 ms, with most requests completing under 1 second.
- Throughput reached 61 requests per second, indicating good performance for moderate concurrency.

## Part IV

[Back to the start](#mle-challenge)

Well, this part was really hard to get done.

I will begin by describing all the things that gone wrong and could have gone so much better:

- First, I spent 2-3 hours trying to properly set the CI-CD github actions. I developed a well wrriten CI-CD workflow, but when I tried to visualize the workflows on Github Actions, they were not recognized by the platform. I tried a lot of changes (like really a lot) to later realizar the reason they were not recognized was the files were not inside `.github/workflows` folder (they were inside `.github`). 

- The previous reason really messed my gitflow practices: on a desperate intent to solve this issue, I thought the reason the workflows were not recognized was the workflows were not on the main branch yet (I was working on a separate branch). So I pushed these changes to develop, and from develop to main. All of this to later realize that was not the solution (like I previously said, the solution was to move the workflows inside the `workflows` folder). 

- Once the workflows were recognized, I tested them to make sure they worked correctly: they did not. So I started working on them. I don't know why, but at that time my mind was still thinking that the workflows only worked on the main branch. So I started developing a hotfix, commit and pushed these changes to develop, and from develop to main. This repeated many times, hotfix after hotfix. All to later realize I could fix and test the workflows on a separate branch, get the workflows working and then merge with develop and main. So yeah, I pushed a lot to those branches (sorry, my bad).

- To make the things worse, getting to make the workflows was SUCH A HUSSLE MAN. As stated before, I deployed my API to GCP, a cloud I do not know much about. The proposed flow was simple:
  - CI:
    - Apply Formater
    - Apply Linter
    - Test code
  - CD:
    - Build image
    - Push image to a container registry (at first, I wanted to use DockerHub)
    - Use the image to deploy a Cloud Run instance

  The CI part was not so difficult: I just required to reapply the same lines of code I was executing locally on the workflow. But man, the CD. The CD was just a completely different story.

  First, I learned settting a connection from DockerHub to GCP was not straightforward, so I replaced DockerHub to GCP Artifact Registry. That shouldn't be so hard, right? Wrong.

  Permissions and roles. I quickly learned I needed to set up some authentication to my workflow in order to push the built image to the repository. So I started reading, watching videos, etc in order to learn all the configurations needed to set this workflow correctly. Man, I learned A LOT. IAM, service account, gcp CLI, etc. I really hope my project passes the evaluation, because I really wouldn't wish this for someone else to have the same experience.

  Another thing that went wrong really bad was the way I was trying to set the steps of both workflows. As I was taught, a good practice is to use Github Actions workflows built by other people (specially the ones built by big companies like Google). So I searched for workflows to deploy Images to GCP and tried to use them. Of course, this went really wrong. None of the workflows worked. I tried many times to fix the workflows, reading on the web, watching a lot of videos. All to later give up on these workflows and switch to `gcloud CLI`, and then, it all started working like a charm. Just magic. But man, I do not joke saying I spent a full day fixing this particular issue.

  <p align="center">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmk2NjB1YTBtcHFmd3g3c29icnZuajNoejd3Ym9zbnN0MWhybXRxbyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/E2USislQIlsfm/giphy.gif" alt="God why" width="400" />
  </p>

Anyways, this is the solution I came up:

- **CI:**
  - Apply Formater (Black)
  - Apply Linter (pylint)
  - Test code (using provided tests)
- **CD:**
  - Build image (Docker)
  - Push image (GCP Artifact Registrry)
  - Deploy image (Cloud Run)

And here's the respective code:

- **Continuous Integration (CI):**

```yaml
name: 'Continuous Integration'

on:
    workflow_call:
    pull_request:
        branches:
            [main, develop]
    push:
        branches:
            [main, develop]

    # Allow manual triggering of the workflow
    workflow_dispatch:

jobs:
    ci:
        runs-on: ubuntu-latest

        steps:
            - name: Checkout code
              uses: actions/checkout@v3

            - name: Set up Python
              uses: actions/setup-python@v4
              with: 
                python-version: "3.10.18"

            - name: Install dependencies
              run: |
                python -m pip install --upgrade pip
                pip install -r requirements-test.txt

            - name: Test App Code
              run: |
                make model-test
                make api-test

            - name: Format (formats code for readability)
              run: |
                black .

            - name: Lint (checks for bad code style)
              run: |
                pylint --disable=R,C  challenge/api.py
```

- **Continuous Delivery (CD):**

```yaml
name: 'Continuous Delivery'
on:
  push:
      branches:
          [main]

  workflow_dispatch:

permissions:
  id-token: write
  contents: read

jobs:
  ci:
    uses: ./.github/workflows/ci.yml
  deploy:
    needs: ci
    runs-on: ubuntu-latest
    env:
      DOCKER_TAG: ${{ secrets.REGION }}-docker.pkg.dev/${{ secrets.PROJECT_ID }}/${{ secrets.SERVICE }}/${{ secrets.IMAGE_NAME }}:${{ github.sha }}
      GOOGLE_PROJECT: ${{ secrets.PROJECT_ID }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Authenticate to Google Cloud
        id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.SERVICE_ACCOUNT }}

      - name: Install Gcloud CLI
        uses: google-github-actions/setup-gcloud@v2
        with:
          project_id: ${{ secrets.PROJECT_ID }}

      - name: Configure Docker for GCP
        run: gcloud auth configure-docker ${{ secrets.REGION }}-docker.pkg.dev

      - name: Build Docker image
        run: docker build --tag "$DOCKER_TAG" .

      - name: Push Docker image
        run: docker push "$DOCKER_TAG"
          
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy mle-challenge \
            --image ${{ env.DOCKER_TAG }} \
            --platform managed \
            --region ${{ secrets.REGION }} \
            --service-account ${{ secrets.SERVICE_ACCOUNT }} \
            --port 8000 \
            --allow-unauthenticated \
            --min-instances 1
```

## Closure

[Back to the start](#mle-challenge)

This is it! If you read through the whole solution, thanks for your attention. I really hope this solution meets the desired outcome to approve the challenge. If not, well at least I had fun.

See you next time!!

<p align="center">
<img src="https://media1.tenor.com/m/Syo75lZhQ4gAAAAC/rambo-thumbs-up-rambo.gif" alt="God why" width="300" />
</p>