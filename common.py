import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE

DATA_PATH = "/content/drive/My Drive/dataset/0.05percent_34classes.csv"
RANDOM_STATE = 42


def prepare_data(path=DATA_PATH, test_size=0.2):
    """Split first; fit preprocessing on training data; apply SMOTE to train only."""
    data = pd.read_csv(path)
    if "label" not in data.columns:
        raise ValueError(
            "Expected a target column named 'label'. "
            f"First columns found: {list(data.columns[:15])}"
        )

    data = data.dropna(subset=["label"]).copy()
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(data["label"].astype(str))
    X = data.drop(columns=["label"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    numeric_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train_raw.select_dtypes(exclude=["number"]).columns.tolist()
    transformers = []

    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="mean"))]),
            numeric_cols
        ))
    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE is applied only to the training set.
    X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train, y_train
    )

    print("Training shape after SMOTE:", X_train.shape)
    print("Untouched test shape:", X_test.shape)
    print("Number of classes:", len(y_encoder.classes_))
    return X_train, X_test, y_train, y_test, y_encoder


def evaluate_model(model, X_test, y_test, encoder, model_name, y_prob=None):
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

    if y_prob is not None:
        try:
            auc_score = roc_auc_score(
                y_test, y_prob, labels=np.arange(len(encoder.classes_)),
                multi_class="ovr", average="macro"
            )
            print(f"Macro OvR ROC-AUC: {auc_score:.4f}")
        except Exception as exc:
            print("ROC-AUC could not be calculated:", exc)

    cm = confusion_matrix(
        y_test, y_pred, labels=np.arange(len(encoder.classes_))
    )
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    output_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(output_name, dpi=200)
    plt.show()
    print(f"Saved confusion matrix: {output_name}")
