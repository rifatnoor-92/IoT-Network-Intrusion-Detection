# Multiclass IoT Network Intrusion Detection

This repository contains implementations of three classifiers using a CICIoT2023-derived CSV dataset:

- Random Forest
- XGBoost
- CNN-GRU

## Methodology implemented
1. Read the CSV and separate the `label` target.
2. Split the data into 80% training and 20% holdout test sets using stratification.
3. Fit numeric mean imputation and categorical most-frequent imputation/one-hot encoding on training data only.
4. Fit `StandardScaler` on training data and transform the test data.
5. Apply SMOTE to training data only.
6. Train the model and evaluate on the untouched test set.

Metrics: accuracy, weighted precision, weighted recall, weighted F1-score, confusion matrix, and macro one-vs-rest ROC-AUC where supported.

## Dataset setup
The default path in `src/common.py` is:

`/content/drive/My Drive/dataset/0.05percent_34classes.csv`

The CSV must contain a target column named `label`. Update `DATA_PATH` in `src/common.py` if needed. Add the official dataset download/source URL before publishing. Do not commit the dataset CSV unless its terms permit redistribution.

## Run in Google Colab
Mount Google Drive in a notebook cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Install requirements:
```python
!pip install -r requirements.txt
```

Run a model from the repository root:
```python
!python src/random_forest.py
!python src/xgboost_model.py
!python src/cnn_gru.py
```

## Reproducibility notes
- This version splits before fitting preprocessing and before SMOTE. Results can differ from earlier experiments that scaled or oversampled the full dataset before splitting.
- CNN-GRU currently uses 10 epochs and a validation split from training data. Align this with the final manuscript only if it matches the experiment actually reported.
- CNN-GRU treats the feature vector as a 1D sequence; explain this modelling choice in the paper.
