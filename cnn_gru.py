import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from common import prepare_data


def main():
    X_train, X_test, y_train, y_test, encoder = prepare_data()
    number_of_features = X_train.shape[1]
    number_of_classes = len(encoder.classes_)

    # Conv1D expects (samples, steps/features, channels).
    X_train_3d = X_train.reshape(X_train.shape[0], number_of_features, 1)
    X_test_3d = X_test.reshape(X_test.shape[0], number_of_features, 1)

    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu",
               input_shape=(number_of_features, 1)),
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

    # Validation split is taken from the training set, not the final test set.
    history = model.fit(
        X_train_3d, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN-GRU Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_gru_training_validation_accuracy.png", dpi=200)
    plt.show()

    probabilities = model.predict(X_test_3d, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\n=== CNN-GRU ===")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=np.arange(number_of_classes),
        target_names=encoder.classes_,
        zero_division=0
    ))

    try:
        auc_score = roc_auc_score(
            y_test, probabilities,
            labels=np.arange(number_of_classes),
            multi_class="ovr", average="macro"
        )
        print(f"Macro OvR ROC-AUC: {auc_score:.4f}")
    except Exception as exc:
        print("ROC-AUC could not be calculated:", exc)

    cm = confusion_matrix(
        y_test, predictions, labels=np.arange(number_of_classes)
    )
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("CNN-GRU Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("cnn_gru_confusion_matrix.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
