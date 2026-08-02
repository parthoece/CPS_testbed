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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from joblib import Parallel, delayed

# ------------------------ Logging Setup ------------------------
output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/log/xgboost"
os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(output_dir, "log_train_xgboost.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------ Load Data ------------------------
data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/log/split"
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
X_val = np.load(os.path.join(data_dir, "X_val.npy"))
y_val = np.load(os.path.join(data_dir, "y_val.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))

logging.info("Loaded dataset successfully.")
y_train = y_train.reshape(-1)

# ------------------------ Class Distribution ------------------------
class_counts = collections.Counter(y_train.tolist())
logging.info(f"Class Distribution: {class_counts}")

# ------------------------ Optuna Objective ------------------------
def objective(trial):
    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "n_estimators": 500,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "random_state": 42,
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")
    return f1

# ------------------------ Run Optuna ------------------------
logging.info("Starting Optuna hyperparameter tuning for XGBoost...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
logging.info("Hyperparameter tuning completed.")

# ------------------------ Create Output Folders ------------------------
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "classification_reports"), exist_ok=True)

# ------------------------ Train and Save All Trials ------------------------
all_metrics = []
all_trial_cms = {}

def train_trial(i, params):
    start_time = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(output_dir, "models", f"xgboost_model_trial_{i}.pkl")
    joblib.dump(model, model_path)

    # Evaluate on test data
    y_pred_test = model.predict(X_test)
    precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=1)
    acc = accuracy_score(y_test, y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    class_report = classification_report(y_test, y_pred_test, zero_division=1, output_dict=True)
    report_path = os.path.join(output_dir, "classification_reports", f"classification_report_trial_{i}.json")
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Trial {i})")
    plt.savefig(os.path.join(output_dir, "confusion_matrices", f"cm_trial_{i}.png"))
    plt.close()

    all_metrics.append({
        "trial": i,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc
    })

    all_trial_cms[f"trial_{i}"] = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_percent": cm_percent.tolist()
    }

    elapsed_time = time.time() - start_time
    logging.info(f"Trial {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

# Train trials in parallel
Parallel(n_jobs=-1)(delayed(train_trial)(i, trial.params) for i, trial in enumerate(study.trials))

pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "xgboost_all_trials_metrics.csv"), index=False)
with open(os.path.join(output_dir, "xgboost_all_trials_cm.json"), "w") as json_file:
    json.dump(all_trial_cms, json_file, indent=4)

logging.info("All trial metrics and confusion matrices saved.")

# ------------------------ Train and Save Best Model ------------------------
logging.info("Training the best XGBoost model...")
best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train, y_train)
joblib.dump(best_model, os.path.join(output_dir, "best_xgboost_model.pkl"))

# Evaluate on test data
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
plt.title("Best Model Confusion Matrix")
plt.savefig(os.path.join(output_dir, "best_xgboost_confusion_matrix.png"))
plt.close()

logging.info("Best model confusion matrix saved.")
print("XGBoost model training, evaluation, and saving completed.")
