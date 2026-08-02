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
import lightgbm as lgb
import os
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

output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/log/lgbm/"
os.makedirs(output_dir, exist_ok=True)

log_file =  "log_train_lgbm.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/log/split"
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
X_val = np.load(os.path.join(data_dir, "X_val.npy"))
y_val = np.load(os.path.join(data_dir, "y_val.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))

logging.info(" Loaded dataset successfully.")
y_train = y_train.reshape(-1)  # Ensure correct shape

class_counts = collections.Counter(y_train.tolist())
logging.info(f"Class Distribution: {class_counts}")

def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256), 
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),  
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-3, 1.0), 
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-3, 1.0),
        "max_bin": trial.suggest_int("max_bin", 128, 512),  
        "scale_pos_weight": trial.suggest_loguniform("scale_pos_weight", 0.1, 10.0) 
    }

    model = lgb.LGBMClassifier(**params, random_state=42, n_estimators=500)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")

    return f1  


logging.info(" Starting Optuna hyperparameter tuning for LightGBM...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
logging.info(" Hyperparameter tuning completed.")

os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "classification_reports"), exist_ok=True)

all_metrics = []
all_trial_cms = {}

def train_trial(i, params):
    start_time = time.time()

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Save model weights
    model_path = os.path.join(output_dir, "models", f"lgbm_model_trial_{i}.pkl")
    joblib.dump(model, model_path)

    # Evaluate model
    y_pred_test = model.predict(X_test)
    precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=1)
    acc = accuracy_score(y_test, y_pred_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Save classification report
    class_report = classification_report(y_test, y_pred_test, zero_division=1, output_dict=True)
    report_path = os.path.join(output_dir, "classification_reports", f"classification_report_trial_{i}.json")
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=4)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Trial {i})")
    plt.savefig(os.path.join(output_dir, "confusion_matrices", f"cm_trial_{i}.png"))
    plt.close()

    all_metrics.append({"trial": i, "precision": precision, "recall": recall, "f1": f1, "accuracy": acc})
    all_trial_cms[f"trial_{i}"] = {"confusion_matrix": cm.tolist(), "confusion_matrix_percent": cm_percent.tolist()}

    elapsed_time = time.time() - start_time
    logging.info(f" Trial {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

#  Parallelize trial training
Parallel(n_jobs=-1)(delayed(train_trial)(i, trial.params) for i, trial in enumerate(study.trials))

#  Save all trials metrics and confusion matrices
pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "lgbm_all_trials_metrics.csv"), index=False)
with open(os.path.join(output_dir, "lgbm_all_trials_cm.json"), "w") as json_file:
    json.dump(all_trial_cms, json_file, indent=4)
logging.info(" All trial metrics and confusion matrices saved.")

logging.info(" Training the best LightGBM model...")
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)
joblib.dump(best_model, os.path.join(output_dir, "best_lgbm_model.pkl"))

y_pred_test = best_model.predict(X_test)
precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=1)
acc = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

logging.info(f" Best Model - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "best_lgbm_confusion_matrix.png"))
plt.close()

logging.info(" Best model confusion matrix saved.")
print(" LightGBM model training, evaluation, and saving completed.")
