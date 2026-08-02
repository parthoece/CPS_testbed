import os
import pandas as pd
import logging
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, f1_score, make_scorer, classification_report
import subprocess
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import joblib
import optuna
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


log_file = "lof_train_traffic.log"

logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f" Starting LOF traffic")


def train_lof(train_data, n_neighbors=20, contamination=0.1):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof.fit(train_data)
    logging.info(f"LOF model trained with n_neighbors={n_neighbors}, contamination={contamination}.")
    return lof


def tune_threshold(model, val_data, val_labels, output_dir, stage, trial_number):
    decision_scores = -model.decision_function(val_data).reshape(-1, 1)
    scaler = MinMaxScaler()
    decision_scores_normalized = scaler.fit_transform(decision_scores).flatten()

    thresholds = np.linspace(0, 1, 100)
    f1_scores, precision_scores, recall_scores, y_pred_binaries = [], [], [], []

    for threshold in thresholds:
        y_pred = np.where(decision_scores_normalized >= threshold, 1, 0)
        f1_scores.append(f1_score(val_labels, y_pred))
        precision_scores.append(precision_score(val_labels, y_pred, zero_division=0))
        recall_scores.append(recall_score(val_labels, y_pred, zero_division=0))
        y_pred_binaries.append(y_pred)

    best_idx = np.argmax(f1_scores)
    cm = confusion_matrix(val_labels, y_pred_binaries[best_idx])
    cm_pct = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100

    logging.info(f"Best Threshold: {thresholds[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}, Precision: {precision_scores[best_idx]:.4f}, Recall: {recall_scores[best_idx]:.4f}")
    logging.info(f"Confusion Matrix (Percentage):\n{np.round(cm_pct, 2)}")

    return thresholds[best_idx], f1_scores[best_idx], precision_scores[best_idx], recall_scores[best_idx], cm_pct


def evaluate_lof(model, test_data, test_labels, threshold):
    scores = -model.decision_function(test_data)
    y_pred = np.where(scores >= threshold, 1, 0)
    f1 = f1_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall = recall_score(test_labels, y_pred, zero_division=0)
    cm = confusion_matrix(test_labels, y_pred)
    cm_pct = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100
    return f1, precision, recall, cm_pct, y_pred


def train_and_validate_stage(train_data, val_data, val_labels, test_data, test_labels, output_dir, stage, n_trials=10):
    os.makedirs(output_dir, exist_ok=True)
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial_output_dir = os.path.join(output_dir, f"stage{stage}_trial_{trial.number}")
        os.makedirs(trial_output_dir, exist_ok=True)
        
        n_neighbors = trial.suggest_int("n_neighbors", 10, 50)
        contamination = trial.suggest_float("contamination", 0.01, 0.2)
        model = train_lof(train_data, n_neighbors=n_neighbors, contamination=contamination)
        joblib.dump(model, os.path.join(trial_output_dir, f"model_stage{stage}_trial{trial.number}.pkl"))
        threshold, best_f1, best_prec, best_rec, _ = tune_threshold(model, val_data, val_labels, output_dir, stage, trial.number)
        score = 0.5 * best_f1 + 0.25 * best_prec + 0.25 * best_rec
        return score

    study.optimize(objective, n_trials=n_trials)
    params = study.best_params
    model = train_lof(train_data, **params)
    threshold, best_f1, best_prec, best_rec, cm_pct = tune_threshold(model, val_data, val_labels, output_dir, stage, study.best_trial.number)
    f1, prec, rec, cm_pct, y_pred = evaluate_lof(model, test_data, test_labels, threshold)
    joblib.dump(model, os.path.join(output_dir, f"stage{stage}_lof_model.pkl"))
    return model, threshold, y_pred


def filter_stage2_data(stage1_model, threshold, val_data, test_data, val_labels, test_labels):
    val_scores = -stage1_model.decision_function(val_data)
    test_scores = -stage1_model.decision_function(test_data)

    val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min())
    test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min())

    val_idx = val_scores >= threshold
    test_idx = test_scores >= threshold

    val_data_filtered = val_data[val_idx]
    test_data_filtered = test_data[test_idx]
    val_labels_filtered = val_labels[val_idx]
    test_labels_filtered = test_labels[test_idx]

    val_labels_filtered[val_labels_filtered == 1] = 0
    val_labels_filtered[val_labels_filtered == 2] = 1
    test_labels_filtered[test_labels_filtered == 1] = 0
    test_labels_filtered[test_labels_filtered == 2] = 1

    return val_data_filtered, test_data_filtered, val_labels_filtered, test_labels_filtered


def main():
    base_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/train/preprocess/traffic/split"
    normal_train_data = np.load(os.path.join(base_dir, "normal_train_minmax_scaled.npy"))
    fault_train_data = np.load(os.path.join(base_dir, "fault_train_minmax_scaled.npy"))
    val_data = np.load(os.path.join(base_dir, "final_validation_minmax_scaled.npy"))
    test_data = np.load(os.path.join(base_dir, "final_test_minmax_scaled.npy"))
    val_labels = np.load(os.path.join(base_dir, "final_validation_labels.npy")).ravel()
    test_labels = np.load(os.path.join(base_dir, "final_test_labels.npy")).ravel()

    val_labels_stage1 = np.where(val_labels > 0, 1, 0)
    test_labels_stage1 = np.where(test_labels > 0, 1, 0)

    output_dir_stage1 = os.path.join(base_dir, "stage1_output_lof")
    output_dir_stage2 = os.path.join(base_dir, "stage2_output_lof")

    best_model_stage1, best_threshold_stage1, stage1_y_pred = train_and_validate_stage(
        train_data=normal_train_data,
        val_data=val_data,
        val_labels=val_labels_stage1,
        test_data=test_data,
        test_labels=test_labels_stage1,
        output_dir=output_dir_stage1,
        stage=1,
        n_trials=10
    )

    val_data_stage2, test_data_stage2, val_labels_stage2, test_labels_stage2 = filter_stage2_data(
        best_model_stage1, best_threshold_stage1, val_data, test_data, val_labels.copy(), test_labels.copy())

    best_model_stage2, best_threshold_stage2, stage2_y_pred = train_and_validate_stage(
        train_data=fault_train_data,
        val_data=val_data_stage2,
        val_labels=val_labels_stage2,
        test_data=test_data_stage2,
        test_labels=test_labels_stage2,
        output_dir=output_dir_stage2,
        stage=2,
        n_trials=10
    )

    logging.info("LOF two-stage anomaly detection pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")