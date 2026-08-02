import os
import logging
import numpy as np
import joblib
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

log_file = "isoforest_train_sensor.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f" Starting isolationforest on sensor data")

def train_isolation_forest(train_data, n_estimators=100, contamination=0.1):
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(train_data)
    return model


def tune_threshold(model, val_data, val_labels, output_dir, stage, trial_number):
    scores = -model.decision_function(val_data)
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    thresholds = np.linspace(0, 1, 100)
    f1_scores, precision_scores, recall_scores, y_preds = [], [], [], []

    for t in thresholds:
        y_pred = np.where(scores >= t, 1, 0)
        f1_scores.append(f1_score(val_labels, y_pred))
        precision_scores.append(precision_score(val_labels, y_pred, zero_division=0))
        recall_scores.append(recall_score(val_labels, y_pred, zero_division=0))
        y_preds.append(y_pred)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    cm = confusion_matrix(val_labels, y_preds[best_idx])
    cm_pct = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100

    logging.info(f"Stage {stage} - Trial {trial_number} - Best Threshold: {best_thresh:.4f}, F1: {f1_scores[best_idx]:.4f}")
    logging.info(f"Confusion Matrix (Percentage):\n{np.round(cm_pct, 2)}")
    return best_thresh, f1_scores[best_idx], precision_scores[best_idx], recall_scores[best_idx], cm_pct


def evaluate_model(model, test_data, test_labels, threshold):
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
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        contamination = trial.suggest_float("contamination", 0.01, 0.2)
        model = train_isolation_forest(train_data, n_estimators, contamination)
        joblib.dump(model, os.path.join(trial_output_dir, f"model_stage{stage}_trial{trial.number}.pkl"))
        threshold, f1, prec, rec, _ = tune_threshold(model, val_data, val_labels, output_dir, stage, trial.number)
        logging.info(f"Stage {stage} - Trial {trial.number} - Params: n_estimators={n_estimators}, contamination={contamination}")
        return 0.5 * f1 + 0.25 * prec + 0.25 * rec

    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    model = train_isolation_forest(train_data, best_params['n_estimators'], best_params['contamination'])
    threshold, f1, prec, rec, cm_pct = tune_threshold(model, val_data, val_labels, output_dir, stage, study.best_trial.number)
    f1, prec, rec, cm_pct, y_pred = evaluate_model(model, test_data, test_labels, threshold)
    joblib.dump(model, os.path.join(output_dir, f"stage{stage}_isoforest_model.pkl"))
    return model, threshold, y_pred


def filter_stage2_data(model, threshold, val_data, test_data, val_labels, test_labels):
    val_scores = -model.decision_function(val_data)
    test_scores = -model.decision_function(test_data)

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
    base_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/train/preprocess/sensor/split_final"
    normal_train_data = np.load(os.path.join(base_dir, "normal_train_oneclass_svm_scaled_data.npy"))
    fault_train_data = np.load(os.path.join(base_dir, "fault_train_oneclass_svm_scaled_data.npy"))
    val_data = np.load(os.path.join(base_dir, "final_validation_oneclass_svm_scaled_data.npy"))
    test_data = np.load(os.path.join(base_dir, "final_test_oneclass_svm_scaled_data.npy"))
    val_labels = np.load(os.path.join(base_dir, "final_validation_labels.npy")).ravel()
    test_labels = np.load(os.path.join(base_dir, "final_test_labels.npy")).ravel()


    val_labels_stage1 = np.where(val_labels > 0, 1, 0)
    test_labels_stage1 = np.where(test_labels > 0, 1, 0)

    output_dir_stage1 = os.path.join(base_dir, "stage1_output_isoforest")
    output_dir_stage2 = os.path.join(base_dir, "stage2_output_isoforest")

    model_stage1, threshold_stage1, pred_stage1 = train_and_validate_stage(
        normal_train_data, val_data, val_labels_stage1, test_data, test_labels_stage1, output_dir_stage1, stage=1
    )

    val_data_stage2, test_data_stage2, val_labels_stage2, test_labels_stage2 = filter_stage2_data(
        model_stage1, threshold_stage1, val_data, test_data, val_labels.copy(), test_labels.copy())

    model_stage2, threshold_stage2, pred_stage2 = train_and_validate_stage(
        fault_train_data, val_data_stage2, val_labels_stage2, test_data_stage2, test_labels_stage2, output_dir_stage2, stage=2
    )

    logging.info("\u2705 IsolationForest-based two-stage anomaly detection pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
