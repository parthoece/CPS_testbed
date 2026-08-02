import os
import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, make_scorer, classification_report
import matplotlib.pyplot as plt
import joblib
import optuna
import subprocess
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score



log_file = "train_traffic_ocsvm3.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Starting OCSVM traffic_full features ")

def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set up at {output_dir}")

def load_data(file_path, dataset_name):
    try:
        data = np.load(file_path)
        logging.info(f"Loaded {dataset_name} data from {file_path}.")
        return data
    except FileNotFoundError as e:
        logging.error(f" File not found: {e}")
        raise

def load_labels(file_path, dataset_name):
    try:
        labels = np.load(file_path)
        logging.info(f"Loaded {dataset_name} labels from {file_path}.")
        return labels.ravel() 
    except FileNotFoundError as e:
        logging.error(f" File not found: {e}")
        raise

def train_one_class_svm(train_data, kernel='rbf', nu=0.01, gamma='scale'):
    try:
        ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        ocsvm.fit(train_data)
        logging.info(f"One-Class SVM model trained with kernel={kernel}, nu={nu}, gamma={gamma}.")
        return ocsvm
    except Exception as e:
        logging.error(f" Error training One-Class SVM: {e}")
        raise



def plot_performance_by_threshold(decision_scores, val_labels, output_dir, stage, trial_number):
    thresholds = np.linspace(min(decision_scores), max(decision_scores), 100)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    y_pred_binaries = []  

    for threshold in thresholds:
        y_pred_binary = np.where(decision_scores >= threshold, 1, 0)
        f1 = f1_score(val_labels, y_pred_binary)
        precision = precision_score(val_labels, y_pred_binary, zero_division=1)
        recall = recall_score(val_labels, y_pred_binary)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        y_pred_binaries.append(y_pred_binary)

    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores[best_threshold_index]
    best_precision = precision_scores[best_threshold_index]
    best_recall = recall_scores[best_threshold_index]
    best_y_pred = y_pred_binaries[best_threshold_index]

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score", color='blue')
    plt.plot(thresholds, precision_scores, label="Precision", color='green')
    plt.plot(thresholds, recall_scores, label="Recall", color='red')
    plt.axvline(x=best_threshold, color='black', linestyle='--', label=f"Best Threshold: {best_threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"F1, Precision, and Recall vs. Threshold (Stage {stage}, Trial {trial_number})")
    plt.legend()
    plt.grid(True)

    f1_plot_path = os.path.join(output_dir, f"f1_precision_recall_by_threshold_stage{stage}_trial{trial_number}.png")
    plt.savefig(f1_plot_path)
    plt.close()

    logging.info(f"F1, Precision, and Recall plot saved to {f1_plot_path}")
    logging.info(f"Best Threshold: {best_threshold:.4f}, F1 Score: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

    return best_threshold, best_f1, best_precision, best_recall



def tune_threshold(model, val_data, val_labels, output_dir, stage, trial_number):
    
    decision_scores = model.decision_function(val_data).reshape(-1, 1)  
    scaler = MinMaxScaler(feature_range=(0, 1))
    decision_scores_normalized = scaler.fit_transform(decision_scores).flatten()

    thresholds = np.linspace(0, 1, 100) 
    f1_scores, precision_scores, recall_scores = [], [], []
    y_pred_binaries = []

    for threshold in thresholds:
        y_pred_binary = np.where(decision_scores_normalized >= threshold, 1, 0)
        f1 = f1_score(val_labels, y_pred_binary)
        precision = precision_score(val_labels, y_pred_binary, zero_division=0)
        recall = recall_score(val_labels, y_pred_binary, zero_division=0)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        y_pred_binaries.append(y_pred_binary)

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]
    best_precision = precision_scores[best_index]
    best_recall = recall_scores[best_index]
    best_y_pred = y_pred_binaries[best_index]

    cm = confusion_matrix(val_labels, best_y_pred)
    cm_percentage = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100  # Normalize row-wise
    
    logging.info(f"Confusion Matrix (Raw Counts) - Stage {stage}, Trial {trial_number}:\n{cm}")
    logging.info(f"Confusion Matrix (Percentage) - Stage {stage}, Trial {trial_number}:\n{np.round(cm_percentage, 2)}%")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Percentage) - Stage {stage}, Trial {trial_number}")

    cm_path = os.path.join(output_dir, f"confusion_matrix_stage{stage}_trial{trial_number}.png")
    plt.savefig(cm_path)
    plt.close()

    logging.info(f"Confusion Matrix (Percentage) saved to {cm_path}")

    return best_threshold, best_f1, best_precision, best_recall, cm_percentage


def evaluate_one_class_svm(model, test_data, test_labels, threshold):
    
    decision_scores = model.decision_function(test_data)
    y_pred_binary = np.where(decision_scores >= threshold, 1, 0) # Anomaly = 0, Normal = 1


    f1 = f1_score(test_labels, y_pred_binary)
    precision = precision_score(test_labels, y_pred_binary, zero_division=0)
    recall = recall_score(test_labels, y_pred_binary, zero_division=0)
    cm = confusion_matrix(test_labels, y_pred_binary)
    cm_percentage = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100  # Convert to percentages

    logging.info(f"One-Class SVM Test Metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    logging.info(f"Confusion Matrix (Raw Counts):\n{cm}")
    logging.info(f"Confusion Matrix (Percentage):\n{np.round(cm_percentage, 2)}%")

    return f1, precision, recall, cm_percentage, y_pred_binary


def create_objective_stage1(train_data, val_data, val_labels, test_data, test_labels, output_dir, all_models_dir, study):

    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
        nu = trial.suggest_float('nu', 0.01, 0.5)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        try:
            ocsvm = train_one_class_svm(train_data, kernel, nu, gamma)

            trial_model_path = os.path.join(all_models_dir, f"stage1_trial_{trial.number}.pkl")
            joblib.dump(ocsvm, trial_model_path)
            logging.info(f"Stage 1 model for trial {trial.number} saved to {trial_model_path}")

            threshold, best_f1, best_precision, best_recall, cm_percentage = tune_threshold(
                ocsvm, val_data, val_labels, output_dir, 1, trial.number
            )

            f1, precision, recall, cm_percentage, y_pred_binary = evaluate_one_class_svm(
                ocsvm, test_data, test_labels, threshold
            )

            logging.info(f"Confusion Matrix (Percentage) - Stage 1, Trial {trial.number}:\n{np.round(cm_percentage, 2)}%")

            score = (0.5 * best_f1) + (0.25 * best_precision) + (0.25 * best_recall)
            
            logging.info(f"Trial {trial.number}: F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Combined Score={score:.4f}")

            trial.set_user_attr("f1_score", best_f1)
            trial.set_user_attr("precision", best_precision)
            trial.set_user_attr("recall", best_recall)

            return score  

        except Exception as e:
            logging.error(f" Trial failed with error: {e}")
            return -1  

    return objective


def create_objective_stage2(train_data, val_data, val_labels, test_data, test_labels, output_dir, all_models_dir, study):

    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
        nu = trial.suggest_float('nu', 0.01, 0.5)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        try:
            ocsvm = train_one_class_svm(train_data, kernel, nu, gamma)

            trial_model_path = os.path.join(all_models_dir, f"stage2_trial_{trial.number}.pkl")
            joblib.dump(ocsvm, trial_model_path)
            logging.info(f"Stage 2 model for trial {trial.number} saved to {trial_model_path}")

            threshold, best_f1, best_precision, best_recall, cm_percentage = tune_threshold(
                ocsvm, val_data, val_labels, output_dir, 2, trial.number
            )

            f1, precision, recall, cm_percentage, y_pred_binary = evaluate_one_class_svm(
                ocsvm, test_data, test_labels, threshold
            )

            logging.info(f"Confusion Matrix (Percentage) - Stage 2, Trial {trial.number}:\n{np.round(cm_percentage, 2)}%")

            score = (0.5 * best_f1) + (0.25 * best_precision) + (0.25 * best_recall)
            
            logging.info(f"Trial {trial.number}: F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Combined Score={score:.4f}")

            trial.set_user_attr("f1_score", best_f1)
            trial.set_user_attr("precision", best_precision)
            trial.set_user_attr("recall", best_recall)

            return score 

        except Exception as e:
            logging.error(f" Trial failed with error: {e}")
            return -1  

    return objective



def filter_and_remap_labels(labels, normal_label=0, anomaly_labels=[1, 2]):
   
    remapped_labels = np.copy(labels)
    for anomaly_label in anomaly_labels:
        remapped_labels[remapped_labels == anomaly_label] = 1
    remapped_labels[remapped_labels == normal_label] = 0
    return remapped_labels

def launch_optuna_dashboard(db_path, port=8091):
    for attempt in range(3): 
        try:
            dashboard_process = subprocess.Popen(["optuna-dashboard", f"sqlite:///{db_path}", "--port", str(port)])
            logging.info(f"Launched Optuna Dashboard on port {port}. Open in your browser.")
            return dashboard_process
        except OSError as e:
            if e.errno == 98: 
                logging.warning(f" Port {port} is already in use. Trying another port...")
                port += 1  
                continue
            else:
                logging.error(f" Error launching Optuna Dashboard: {e}")
                break  
        except FileNotFoundError:
            logging.warning(" Optuna Dashboard is not installed. Install with `pip install optuna-dashboard` to view the dashboard.")
            break  

    logging.error(" Failed to launch Optuna Dashboard after multiple attempts.")
    return None



def train_and_validate_stage1(normal_train, combined_val, combined_test, output_dir, n_trials=10):
    
    all_models_dir = os.path.join(output_dir, "stage1_all_models")
    os.makedirs(all_models_dir, exist_ok=True)
    logging.info(f"Created directory for all Stage 1 models: {all_models_dir}")

    # Load data
    normal_train_data = load_data(normal_train, "Normal Train Data")
    combined_val_data = load_data(combined_val, "Combined Validation Data")
    combined_test_data = load_data(combined_test, "Combined Test Data")

    val_labels = load_labels("/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/traffic/split/final_validation_labels.npy", "Validation Labels")
    test_labels = load_labels("/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/traffic/split/final_test_labels.npy", "Test Labels")

    remapped_val_labels = filter_and_remap_labels(val_labels)
    remapped_test_labels = filter_and_remap_labels(test_labels)

    normal_train_scaled = normal_train_data
    combined_val_scaled = combined_val_data
    combined_test_scaled = combined_test_data

    db_path = os.path.join(output_dir, "optuna_stage1.db")  
    study = optuna.create_study(direction='maximize', storage=f"sqlite:///{db_path}", study_name="Stage1", load_if_exists=True)
    objective = create_objective_stage1(normal_train_scaled, combined_val_scaled, remapped_val_labels, combined_test_scaled, remapped_test_labels, output_dir, all_models_dir, study)

    dashboard_process = launch_optuna_dashboard(db_path)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logging.info(f"Best hyperparameters for Stage 1: {best_params}")

    best_ocsvm = train_one_class_svm(normal_train_scaled, best_params['kernel'], best_params['nu'], best_params['gamma'])

    best_threshold, best_f1, best_precision, best_recall, cm_percentage = tune_threshold(best_ocsvm, combined_val_scaled, remapped_val_labels, output_dir, 1, study.best_trial.number)

    try:
        study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_results_stage1.csv"), index=False)
        logging.info(f"Optuna results for Stage 1 saved to CSV.")
    except Exception as e:
        logging.error(f" Failed to save Optuna results to CSV: {e}")

    best_f1, best_precision, best_recall, cm_percentage, best_y_pred = evaluate_one_class_svm(best_ocsvm, combined_test_scaled, remapped_test_labels, best_threshold)

    best_model_path = os.path.join(output_dir, "stage1_oneclass_svm_model_traffic.pkl")
    joblib.dump(best_ocsvm, best_model_path)
    logging.info(f"One-Class SVM model saved to {best_model_path}")
    
    metrics_path = os.path.join(output_dir, "stage1_evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {best_precision:.4f}\n")
        f.write(f"Recall: {best_recall:.4f}\n\n")

        f.write("Confusion Matrix (Percentage):\n")
        for row in cm_percentage:
            f.write("  ".join(f"{val:.2f}%" for val in row) + "\n")

    logging.info(f"Evaluation metrics saved to {metrics_path}")
    logging.info(f"Confusion Matrix (Percentage) saved in {metrics_path}")

    cm_str = "\n".join(["  ".join(f"{val:.2f}%" for val in row) for row in cm_percentage])
    logging.info(f"Confusion Matrix (Percentage) - Stage 1:\n{cm_str}")

    cm_path = os.path.join(output_dir, "stage1_confusion_matrix_percentage.png")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Percentage) - Stage 1")
    plt.savefig(cm_path)
    plt.close()

    logging.info(f"Confusion Matrix (Percentage) saved to {cm_path}")
    logging.info(f"Stage 1 F1 Score: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

    if dashboard_process:
        dashboard_process.terminate()

    return best_ocsvm, best_threshold, best_y_pred



def train_and_validate_stage2(fault_train, combined_val, combined_test, stage1_model, stage1_threshold, stage1_y_pred, output_dir, n_trials=10):
    
    all_models_dir = os.path.join(output_dir, "stage2_all_models")
    os.makedirs(all_models_dir, exist_ok=True)
    logging.info(f"Created directory for all Stage 2 models: {all_models_dir}")

    fault_train_data = load_data(fault_train, "Fault Train Data")
    combined_val_data = load_data(combined_val, "Combined Validation Data")
    combined_test_data = load_data(combined_test, "Combined Test Data")

    val_labels = load_labels("/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/traffic/split/final_validation_labels.npy", "Validation Labels")
    test_labels = load_labels("/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/traffic/split/final_test_labels.npy", "Test Labels")

    stage1_val_decision_scores = stage1_model.decision_function(combined_val_data)
    stage1_test_decision_scores = stage1_model.decision_function(combined_test_data)

    stage1_val_decision_scores = (stage1_val_decision_scores - np.min(stage1_val_decision_scores)) / (np.max(stage1_val_decision_scores) - np.min(stage1_val_decision_scores))
    stage1_test_decision_scores = (stage1_test_decision_scores - np.min(stage1_test_decision_scores)) / (np.max(stage1_test_decision_scores) - np.min(stage1_test_decision_scores))

    stage1_anomalies_val_indices = stage1_val_decision_scores >= stage1_threshold
    stage1_anomalies_test_indices = stage1_test_decision_scores >= stage1_threshold

    filtered_val_data = combined_val_data[stage1_anomalies_val_indices]
    filtered_test_data = combined_test_data[stage1_anomalies_test_indices]

    filtered_val_labels = val_labels[stage1_anomalies_val_indices]
    filtered_test_labels = test_labels[stage1_anomalies_test_indices]

    filtered_val_labels[filtered_val_labels == 1] = 0  
    filtered_val_labels[filtered_val_labels == 2] = 1  
    filtered_test_labels[filtered_test_labels == 1] = 0
    filtered_test_labels[filtered_test_labels == 2] = 1

    fault_train_scaled = fault_train_data
    filtered_val_scaled = filtered_val_data
    filtered_test_scaled = filtered_test_data

    db_path = os.path.join(output_dir, "optuna_stage2.db")
    study = optuna.create_study(direction='maximize', storage=f"sqlite:///{db_path}", study_name="Stage2", load_if_exists=True)
    objective = create_objective_stage2(fault_train_scaled, filtered_val_scaled, filtered_val_labels, filtered_test_scaled, filtered_test_labels, output_dir, all_models_dir, study)

    dashboard_process = launch_optuna_dashboard(db_path)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logging.info(f"Best hyperparameters for Stage 2: {best_params}")

    best_ocsvm = train_one_class_svm(fault_train_scaled, best_params['kernel'], best_params['nu'], best_params['gamma'])

    best_threshold, best_f1, best_precision, best_recall, cm_percentage = tune_threshold(best_ocsvm, filtered_val_scaled, filtered_val_labels, output_dir, 2, study.best_trial.number)

    try:
        study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_results_stage2.csv"), index=False)
        logging.info(f"Optuna results for Stage 2 saved to CSV.")
    except Exception as e:
        logging.error(f" Failed to save Optuna results to CSV: {e}")

    best_f1, best_precision, best_recall, cm_percentage, _ = evaluate_one_class_svm(
        best_ocsvm, filtered_test_scaled, filtered_test_labels, best_threshold
    )

    best_model_path = os.path.join(output_dir, "stage2_oneclass_svm_model_traffic.pkl")
    joblib.dump(best_ocsvm, best_model_path)
    logging.info(f"Stage 2 One-Class SVM model saved to {best_model_path}")

    metrics_path = os.path.join(output_dir, "stage2_evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {best_precision:.4f}\n")
        f.write(f"Recall: {best_recall:.4f}\n\n")

        f.write("Confusion Matrix (Percentage):\n")
        for row in cm_percentage:
            f.write("  ".join(f"{val:.2f}%" for val in row) + "\n")

    logging.info(f"Evaluation metrics saved to {metrics_path}")
    logging.info(f"Confusion Matrix (Percentage) saved in {metrics_path}")

    cm_str = "\n".join(["  ".join(f"{val:.2f}%" for val in row) for row in cm_percentage])
    logging.info(f"Confusion Matrix (Percentage) - Stage 2:\n{cm_str}")

    cm_path = os.path.join(output_dir, "stage2_confusion_matrix_percentage.png")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Percentage) - Stage 2")
    plt.savefig(cm_path)
    plt.close()

    logging.info(f"Confusion Matrix (Percentage) saved to {cm_path}")
    logging.info(f"Stage 2 F1 Score: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

    if dashboard_process:
        dashboard_process.terminate()

    return best_ocsvm, best_threshold, _


def main_pipeline():

    base_output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/"
    traffic_output_dir = os.path.join(base_output_dir, "traffic/split")

    stage1_output_dir = os.path.join(traffic_output_dir, "stage1_output")
    stage2_output_dir = os.path.join(traffic_output_dir, "stage2_output")
    setup_output_dir(stage1_output_dir)
    setup_output_dir(stage2_output_dir)

    normal_train = os.path.join(traffic_output_dir, "normal_train_minmax_scaled.npy")
    fault_train = os.path.join(traffic_output_dir, "fault_train_minmax_scaled.npy")
    combined_val = os.path.join(traffic_output_dir, "final_validation_minmax_scaled.npy")
    combined_test = os.path.join(traffic_output_dir, "final_test_minmax_scaled.npy")

    stage1_model, stage1_threshold, stage1_y_pred = train_and_validate_stage1(normal_train, combined_val, combined_test, stage1_output_dir , n_trials=30)

    train_and_validate_stage2(fault_train, combined_val, combined_test, stage1_model, stage1_threshold, stage1_y_pred, stage2_output_dir , n_trials=30)

    logging.info("Two-stage training and hyperparameter tuning pipeline completed.")

if __name__ == "__main__":
    try:
        main_pipeline()
    except Exception as e:
        logging.error(f" An error occurred: {str(e)}")
