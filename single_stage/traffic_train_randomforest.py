import numpy as np
import pandas as pd
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import collections
import json
import time
import os
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/traffic/randomforest/"
os.makedirs(output_dir, exist_ok=True)

log_file = "traffic_train_randomforest.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/traffic/split"
X_train = np.load(f"{data_dir}/svm_train_scaled.npy")[:, :-1]
y_train = np.load(f"{data_dir}/svm_train_labels.npy").reshape(-1)
X_val = np.load(f"{data_dir}/final_validation_scaled.npy")[:, :-1]
y_val = np.load(f"{data_dir}/final_validation_labels.npy")
X_test = np.load(f"{data_dir}/final_test_scaled.npy")[:, :-1]
y_test = np.load(f"{data_dir}/final_test_labels.npy")

logging.info("Loaded dataset successfully.")
logging.info(f"Class Distribution: {collections.Counter(y_train.tolist())}")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted", zero_division=1)

logging.info("Starting Optuna hyperparameter tuning for Random Forest...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
logging.info("Hyperparameter tuning completed.")

# Create output directories
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "classification_reports"), exist_ok=True)

all_metrics = []
all_trial_cms = {}

def train_trial(i, params):
    start_time = time.time()

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    model_path = os.path.join(output_dir, "models", f"randomforest_model_trial_{i}.pkl")
    joblib.dump(model, model_path)

    y_pred_test = model.predict(X_test)
    precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=1)
    acc = accuracy_score(y_test, y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    report_path = os.path.join(output_dir, "classification_reports", f"classification_report_trial_{i}.json")
    with open(report_path, "w") as f:
        json.dump(classification_report(y_test, y_pred_test, output_dict=True, zero_division=1), f, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Trial {i})")
    plt.savefig(os.path.join(output_dir, "confusion_matrices", f"cm_trial_{i}.png"))
    plt.close()

    all_metrics.append({"trial": i, "precision": precision, "recall": recall, "f1": f1, "accuracy": acc})
    all_trial_cms[f"trial_{i}"] = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_percent": cm_percent.tolist()
    }

    elapsed_time = time.time() - start_time
    logging.info(f"Trial {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

Parallel(n_jobs=-1)(delayed(train_trial)(i, trial.params) for i, trial in enumerate(study.trials))

pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "randomforest_all_trials_metrics.csv"), index=False)
with open(os.path.join(output_dir, "randomforest_all_trials_cm.json"), "w") as f:
    json.dump(all_trial_cms, f, indent=4)
logging.info("All trial metrics and confusion matrices saved.")

logging.info("Training the best Random Forest model...")
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
joblib.dump(best_model, os.path.join(output_dir, "best_randomforest_model.pkl"))

y_pred_test = best_model.predict(X_test)
precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=1)
acc = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

logging.info(f"Best Model - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "best_randomforest_confusion_matrix.png"))
plt.close()

logging.info("Best model confusion matrix saved.")
print("Random Forest model training, evaluation, and saving completed.")
