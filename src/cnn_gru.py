
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)

from common import prepare_data


def main():
    X_train, X_test, y_train, y_test, encoder = prepare_data()

    number_of_features = X_train.shape[1]
    number_of_classes = len(encoder.classes_)

    # Reshape data for CNN-GRU
    X_train_3d = X_train.reshape(
        X_train.shape[0], number_of_features, 1
    )
    X_test_3d = X_test.reshape(
        X_test.shape[0], number_of_features, 1
    )

    # Build CNN-GRU model
    model = Sequential([
        Conv1D(
            64,
            kernel_size=3,
            activation="relu",
            input_shape=(number_of_features, 1)
        ),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        GRU(128, return_sequences=False),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(number_of_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train the model
    history = model.fit(
        X_train_3d,
        y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    # Print final training and validation accuracy
    final_train_accuracy = history.history["accuracy"][-1]
    final_val_accuracy = history.history["val_accuracy"][-1]

    print(f"\nTraining Accuracy:   {final_train_accuracy:.4f}")
    print(f"Validation Accuracy: {final_val_accuracy:.4f}")

    # Print best validation accuracy
    best_val_accuracy = max(history.history["val_accuracy"])
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN-GRU Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "cnn_gru_training_validation_accuracy.png",
        dpi=200
    )
    plt.show()

    # Evaluate on the untouched test set
    probabilities = model.predict(X_test_3d, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\n=== CNN-GRU Test Results ===")

    print(
        f"Test Accuracy:  "
        f"{accuracy_score(y_test, predictions):.4f}"
    )
    print(
        f"Precision:     "
        f"{precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall:        "
        f"{recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1-score:      "
        f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )

    # Classification report
    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(number_of_classes),
        target_names=encoder.classes_,
        zero_division=0
    ))

    # Multiclass ROC-AUC
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
