from sklearn.ensemble import RandomForestClassifier
from common import prepare_data, evaluate_model, RANDOM_STATE

def main():
    X_train, X_test, y_train, y_test, encoder = prepare_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)

    evaluate_model(
        model, X_test, y_test, encoder,
        model_name="Random Forest",
        y_prob=probabilities
    )

if __name__ == "__main__":
    main()
