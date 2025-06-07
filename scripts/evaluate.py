import argparse
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(data_path, model_path='models/knn_model.pkl', output_path='output'):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Load model
    model = joblib.load(model_path)
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save report
    report_path = os.path.join(output_path, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Accuracy: {:.4f}\n".format(acc))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print("✅ Evaluation complete. Report saved to:", report_path)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    print("📊 Confusion matrix saved to:", cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained KNN model for wine quality prediction.")
    parser.add_argument("--data_path", type=str, default="data/winequality.csv", help="Path to the CSV dataset")
    parser.add_argument("--model_path", type=str, default="models/knn_model.pkl", help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save evaluation results")

    args = parser.parse_args()

    evaluate_model(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path
    )
