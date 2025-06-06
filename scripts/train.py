
import argparse
import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path):
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X, y

def create_pipeline(n_neighbors):
    logging.info(f"Creating pipeline with KNeighborsClassifier(n_neighbors={n_neighbors})")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    return pipeline

def evaluate_model(model, X_test, y_test, output_path):
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs(output_path, exist_ok=True)
    report_path = os.path.join(output_path, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Accuracy: {:.4f}\n".format(acc))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    logging.info(f"Evaluation report saved to {report_path}")
    logging.info("\n" + report)

def perform_grid_search(X_train, y_train):
    logging.info("Starting GridSearchCV for hyperparameter tuning...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    param_grid = {
        "knn__n_neighbors": list(range(1, 31)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid.best_params_}")
    return grid.best_estimator_

def main(args):
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    if args.use_grid_search:
        model = perform_grid_search(X_train, y_train)
    else:
        model = create_pipeline(n_neighbors=args.n_neighbors)
        model.fit(X_train, y_train)

    logging.info("Saving model...")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model, args.model_path)
    logging.info(f"Model saved to {args.model_path}")

    evaluate_model(model, X_test, y_test, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a KNN model for wine quality prediction.")
    parser.add_argument("--data_path", type=str, default="data/winequality.csv", help="Path to the CSV dataset")
    parser.add_argument("--model_path", type=str, default="models/knn_model.pkl", help="Path to save the trained model")
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save evaluation report")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors (ignored if --use_grid_search is set)")
    parser.add_argument("--use_grid_search", action="store_true", help="Enable GridSearchCV for hyperparameter tuning")

    args = parser.parse_args()
    main(args)
