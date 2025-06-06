
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(data_path, model_path='models/knn_model.pkl', output_path='output'):
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    os.makedirs(output_path, exist_ok=True)
    report_path = os.path.join(output_path, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Accuracy: {:.4f}\n".format(acc))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print("Evaluation complete. Report saved to:", report_path)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    print("Confusion matrix saved to:", cm_path)

if __name__ == "__main__":
    evaluate_model()
