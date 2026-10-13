
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)

from common import prepare_data, RANDOM_STATE


def main():
    X_train, X_test, y_train, y_test, encoder = prepare_data()

    number_of_classes = len(encoder.classes_)

    # Create a validation set from the training data
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    # Initialize XGBoost
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=number_of_classes,
        eval_metric="mlogloss",
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train_fit, y_train_fit)

    # Predictions
    train_predictions = model.predict(X_train_fit)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    # Accuracy scores
    print("\n=== XGBoost ===")

    print(
        f"Training Accuracy:   "
        f"{accuracy_score(y_train_fit, train_predictions):.4f}"
    )
    print(
        f"Validation Accuracy: "
        f"{accuracy_score(y_val, val_predictions):.4f}"
    )
    print(
        f"Test Accuracy:       "
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )

    # Test-set metrics
    print(
        f"Precision: "
        f"{precision_score(y_test, test_predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, test_predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1-score:  "
        f"{f1_score(y_test, test_predictions, average='weighted', zero_division=0):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=np.arange(number_of_classes),
        target_names=encoder.classes_,
        zero_division=0
    ))

    # Multiclass ROC-AUC
    probabilities = model.predict_proba(X_test)

    try:
        auc_score = roc_auc_score(
            y_test,
            probabilities,
            labels=np.arange(number_of_classes),
            multi_class="ovr",
            average="macro"
        )
        print(f"Macro OvR ROC-AUC: {auc_score:.4f}")

    except Exception as exc:
        print("ROC-AUC could not be calculated:", exc)


if __name__ == "__main__":
    main()
