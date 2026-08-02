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
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/traffic/xgboost/"
os.makedirs(output_dir, exist_ok=True)

log_file = "traffic_train_xgboost.log"
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
        "objective": "multi:softmax",
        "num_class": 3,
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "use_label_encoder": False,
        "eval_metric": "mlogloss"
    }

    model = XGBClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted", zero_division=1)

logging.info("Starting Optuna hyperparameter tuning for XGBoost...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
logging.info("Hyperparameter tuning completed.")

os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "classification_reports"), exist_ok=True)

all_metrics = []
all_trial_cms = {}

def train_trial(i, params):
    start_time = time.time()

    model = XGBClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(output_dir, "models", f"xgboost_model_trial_{i}.pkl")
    joblib.dump(model, model_path)

    # Evaluate
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

# Parallel training
Parallel(n_jobs=-1)(delayed(train_trial)(i, trial.params) for i, trial in enumerate(study.trials))

pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "xgboost_all_trials_metrics.csv"), index=False)
with open(os.path.join(output_dir, "xgboost_all_trials_cm.json"), "w") as f:
    json.dump(all_trial_cms, f, indent=4)
logging.info("All trial metrics and confusion matrices saved.")

logging.info("Training the best XGBoost model...")
best_params = study.best_params
best_params.update({
    "objective": "multi:softmax",
    "num_class": 3,
    "use_label_encoder": False,
    "eval_metric": "mlogloss"
})
best_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
joblib.dump(best_model, os.path.join(output_dir, "best_xgboost_model.pkl"))

y_pred_test = best_model.predict(X_test)
precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=1)
acc = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

# Log best model performance
logging.info(f"Best Model - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "best_xgboost_confusion_matrix.png"))
plt.close()

logging.info("Best model confusion matrix saved.")
print("XGBoost model training, evaluation, and saving completed.")
