import os
import numpy as np
import pandas as pd
import logging
import joblib
from collections import Counter
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    filename="timewindow_2stage.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logging.info("Starting...")
data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/ensemble/data"
model_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/ensemble"
output_dir = os.path.join(model_dir, "output/")
os.makedirs(output_dir, exist_ok=True)

sensor_timestamps = pd.read_csv(os.path.join(data_dir, "final_test_timestamps_sensor.csv"))
log_timestamps = pd.read_csv(os.path.join(data_dir, "log_Timestamps_Only.csv"))
traffic_timestamps = pd.read_csv(os.path.join(data_dir, "final_test_timestamps_traffic.csv"))

log_timestamps.rename(columns={"formatted_timestamp": "timestamp"}, inplace=True)
traffic_timestamps.rename(columns={"timestamp": "timestamp"}, inplace=True)

sensor_timestamps["timestamp"] = pd.to_datetime(sensor_timestamps["timestamp"])
log_timestamps["timestamp"] = pd.to_datetime(log_timestamps["timestamp"])
traffic_timestamps["timestamp"] = pd.to_datetime(traffic_timestamps["timestamp"])

sensor_features = np.load(os.path.join(data_dir, "final_test_oneclass_svm_scaled_data_sensor.npy"))
log_features = np.load(os.path.join(data_dir, "test_log.npy"))
traffic_features = np.load(os.path.join(data_dir, "final_test_oneclass_svm_scaled_traffic.npy"))

sensor_labels = np.load(os.path.join(data_dir, "final_test_labels_sensor.npy"))
log_labels = np.load(os.path.join(data_dir, "test_labels_log.npy"))
traffic_labels = np.load(os.path.join(data_dir, "final_test_labels_traffic.npy"))

sensor_timestamps = sensor_timestamps.iloc[:len(sensor_features)]
log_timestamps = log_timestamps.iloc[:len(log_features)]
traffic_timestamps = traffic_timestamps.iloc[:len(traffic_features)]
all_timestamps = pd.concat([sensor_timestamps, log_timestamps, traffic_timestamps]) \
    .drop_duplicates().sort_values("timestamp").reset_index(drop=True)

def align_data(reference_timestamps, timestamps_df, data):
    timestamps_df = timestamps_df.sort_values("timestamp").reset_index(drop=True)
    reference_timestamps = reference_timestamps.sort_values("timestamp").reset_index(drop=True)
    if data.ndim == 1: 
        df = pd.DataFrame({"timestamp": timestamps_df["timestamp"], "label": data.flatten()})
    else: 
        df = pd.DataFrame({"timestamp": timestamps_df["timestamp"]})
        for i in range(data.shape[1]):  
            df[f"feature_{i}"] = data[:, i]

    aligned_df = pd.merge_asof(
        reference_timestamps, df, on="timestamp",
        tolerance=pd.Timedelta(seconds=60), direction="nearest"
    )
    aligned_df.fillna(0, inplace=True)
    return aligned_df


def ensure_feature_match(aligned_data, model, name):
    expected_features = model.n_features_in_  
    actual_features = aligned_data.shape[1]  

    if actual_features != expected_features:
        if actual_features > expected_features:
            aligned_data = aligned_data[:, :expected_features]
        elif actual_features < expected_features:
            raise ValueError
    return aligned_data

def apply_3c(timestamps, predictions, name, size=60, priority_order=[2, 1, 0], ratio=0.75):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps["timestamp"]),
        "prediction": predictions
    })
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    output = []
    start = df.index.min()
    end = df.index.max()

    while start <= end:
        window = df[(df.index >= start) & (df.index < start + pd.Timedelta(seconds=size))]

        if window.empty:
            start += pd.Timedelta(seconds=size)
            continue

        counts = Counter(window["prediction"])
        total = sum(counts.values())
        fault = counts.get(1, 0)
        attack = counts.get(2, 0)

        if name in ["Sensor", "Log", "Traffic"]:
            label = 0
            for cls in priority_order:
                if (window["prediction"] == cls).any():
                    label = cls
                    break
        else:
            if attack > 0:
                attack_ratio = attack / total
                fault_ratio = fault / total
                if attack_ratio >= ratio:
                    label = 2  
                elif fault > attack:
                    label = 1
                elif fault > 0:
                    label = 1  
                else:
                    label = 2
            elif fault > 0:
                label = 1
            else:
                label = 0

        output.extend([label] * len(window))
        start += pd.Timedelta(seconds=size)

    return np.array(output[:len(df)])

stage1_models = {
    "Sensor": joblib.load(os.path.join(model_dir, "stage1_models/stage1_oneclass_svm_model_sensor.pkl")),
    "Log": joblib.load(os.path.join(model_dir, "stage1_models/stage1_oneclass_svm_model_log.pkl")),
    "Traffic": joblib.load(os.path.join(model_dir, "stage1_models/stage1_oneclass_svm_model_traffic.pkl")),
}
stage2_models = {
    "Sensor": joblib.load(os.path.join(model_dir, "stage2_models/stage2_oneclass_svm_model_sensor.pkl")),
    "Log": joblib.load(os.path.join(model_dir, "stage2_models/stage2_oneclass_svm_model_log.pkl")),
    "Traffic": joblib.load(os.path.join(model_dir, "stage2_models/stage2_oneclass_svm_model_traffic.pkl")),
}

aligned_sensor_features = align_data(all_timestamps, sensor_timestamps, sensor_features).drop(columns=["timestamp"]).values
aligned_log_features = align_data(all_timestamps, log_timestamps, log_features).drop(columns=["timestamp"]).values
aligned_traffic_features = align_data(all_timestamps, traffic_timestamps, traffic_features).drop(columns=["timestamp"]).values

aligned_sensor_features = ensure_feature_match(aligned_sensor_features, stage1_models["Sensor"], "Sensor")
aligned_log_features = ensure_feature_match(aligned_log_features, stage1_models["Log"], "Log")
aligned_traffic_features = ensure_feature_match(aligned_traffic_features, stage1_models["Traffic"], "Traffic")

combinations = {
     "Sensor + Log + Traffic": ["Sensor", "Log", "Traffic"]
}

ground_truth_files = {
     "Sensor": "final_test_labels_sensor.npy",
    "Log": "test_labels_log.npy",
    "Traffic": "final_test_labels_traffic.npy",
    "Sensor + Log + Traffic": "aligned_sensor_log_traffic_ground_truth.npy",
    "Sensor + Log": "aligned_sensor_log_ground_truth.npy",
    "Sensor + Traffic": "aligned_sensor_traffic_ground_truth.npy",
    "Log + Traffic": "aligned_traffic_log_ground_truth.npy",
   
}


def convert_labels_stage1(y_true):
    return np.where(y_true > 0, 1, 0)  

def convert_labels_stage2(y_true):
    return np.where(y_true == 1, 0, 1)  

def evaluate_model(y_pred, y_true, stage, output_filename):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=1)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=1)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = np.round(cm.astype(np.float64) / cm.sum(axis=1, keepdims=True) * 100, 2)
    labels = ["Normal", "Anomaly"] if stage == 1 else ["Fault", "Attack"]
    cm_dir = os.path.join(output_dir, "cm/3/th3v2")  
    os.makedirs(cm_dir, exist_ok=True)  
    cm_path = os.path.join(cm_dir, os.path.basename(output_filename))
    plot_confusion_matrix(cm_percent, labels, f"Confusion Matrix (Stage {stage})", cm_path)
    return acc, prec, rec, f1, cm, class_report, cm_percent

def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


results = []
metrics_dict = {}
overall_results = []
for name, sources in combinations.items():
    logging.info(f"Testing combination: {name}")
    combination_metrics = {}
    detection_df = pd.DataFrame({"timestamp": all_timestamps["timestamp"]})
    ground_truth_path = os.path.join(data_dir, ground_truth_files.get(name, ""))
    
    if os.path.exists(ground_truth_path):
        ground_truth_labels = np.load(ground_truth_path)
        if len(sources) == 1: 
            ground_truth_labels = ground_truth_labels[:len(eval(f"{sources[0].lower()}_features"))]
            detection_df = detection_df.iloc[:len(ground_truth_labels)]  
        else:
            min_length = min(len(ground_truth_labels), len(detection_df))
            ground_truth_labels = ground_truth_labels[:min_length]
            detection_df = detection_df.iloc[:min_length]
    else:
        logging.error(f"Ground truth file not found for {name}. Skipping...")
        continue

    threshold_stage1 = {
    "Sensor": 0.80,  
    "Log": 0.35,     
    "Traffic": 0.1 
    }

    predictions_stage1 = []
    for source in sources:
        if len(sources) == 1:  
            X_test = eval(f"{source.lower()}_features")
        else:  
            X_test = eval(f"aligned_{source.lower()}_features")

        stage1_model = stage1_models[source]
        decision_scores = stage1_model.decision_function(X_test)
        threshold = threshold_stage1[source]
        predictions_stage1.append(np.where(decision_scores >= threshold, 1, 0))
    detection_df["Stage1_Final"] = np.any(predictions_stage1, axis=0).astype(int)
    detection_df["GroundTruth_Stage1"] = ground_truth_labels
    detection_df["GroundTruth_Stage1_Binary"] = convert_labels_stage1(ground_truth_labels)
    stage1_anomaly_indices = detection_df["GroundTruth_Stage1_Binary"] != 0
    detection_df["GroundTruth_Stage2"] = convert_labels_stage2(detection_df["GroundTruth_Stage1"])
    cm_filename_stage1 = os.path.join(output_dir, "cm/3/th3v2", f"{name.replace(' ', '_').lower()}_cm_stage1.png")
    os.makedirs(os.path.dirname(cm_filename_stage1), exist_ok=True)  
    acc1, prec1, rec1, f1_1, cm1, class_report1, cm_percent1 = evaluate_model(detection_df["Stage1_Final"], detection_df["GroundTruth_Stage1_Binary"], stage=1, output_filename=cm_filename_stage1)
   
   
    
    combination_metrics['Stage1'] = {
        "Stage1_Thresholds": threshold_stage1,
        'Accuracy': acc1,
        'Precision': prec1,
        'Recall': rec1,
        'F1 Score': f1_1,
        'Confusion Matrix': cm1.tolist(),
        'Confusion Matrix (Percent)': cm_percent1.tolist(),
        'Classification Report': class_report1
    }

    threshold_stage2 = {
    "Sensor": 0.7168,  
    "Log": 0.4004, 
    "Traffic": 0.1270 
    }

    predictions_stage2 = []
    for source in sources:
        if len(sources) == 1: 
            X_test_stage2 = eval(f"{source.lower()}_features")[stage1_anomaly_indices]
        else: 
            X_test_stage2 = eval(f"aligned_{source.lower()}_features")[stage1_anomaly_indices]

        stage2_model = stage2_models[source]
        if len(X_test_stage2) > 0:
            decision_scores_stage2 = stage2_model.decision_function(X_test_stage2)
            threshold = threshold_stage2[source]
            predictions_stage2.append(np.where(decision_scores_stage2 >= threshold, 1, 0))

    if predictions_stage2:
        final_predictions = np.any(predictions_stage2, axis=0).astype(int) 
        detection_df.loc[stage1_anomaly_indices, "Stage2_Final"] = final_predictions
        detection_df["Stage2_Final"] = detection_df["Stage2_Final"].replace({0: 1, 1: 2})  
        cm_filename_stage2 = os.path.join(output_dir, "cm/3/th3v2", f"{name.replace(' ', '_').lower()}_cm_stage2.png")
        acc2, prec2, rec2, f1_2, cm2,class_report2, cm_percent2 = evaluate_model(final_predictions, detection_df.loc[stage1_anomaly_indices, "GroundTruth_Stage2"], stage=2, output_filename=cm_filename_stage2)

        combination_metrics['Stage2'] = {
            "Stage2_Thresholds": threshold_stage2,
            'Accuracy': acc2,
            'Precision': prec2,
            'Recall': rec2,
            'F1 Score': f1_2,
            'Confusion Matrix': cm2.tolist(),
            'Confusion Matrix (Percent)': cm_percent2.tolist(),
            'Classification Report': class_report2
        }

    else:
        logging.info(f"No Stage 2 anomalies detected for {name}")

    logging.info(f"Performing Final 3-Class Evaluation for {name}...")
    final_predictions = np.zeros(len(detection_df), dtype=int)
    final_predictions[detection_df["Stage1_Final"] == 1] = -1  

    if "Stage2_Final" in detection_df.columns:
        detection_df["Stage2_Final"] = detection_df["Stage2_Final"].fillna(0).astype(int)
        stage2_indices = detection_df.loc[stage1_anomaly_indices].index
        final_predictions[stage2_indices] = detection_df.loc[stage2_indices, "Stage2_Final"].replace({0: 1})

    valid_labels = {0, 1, 2, -1}  
    if not set(np.unique(final_predictions)).issubset(valid_labels):
        final_predictions = np.clip(final_predictions, -1, 2)  

    final_predictions[final_predictions == -1] = 0  
    final_ground_truth = detection_df["GroundTruth_Stage1"]  
    
    f1_window_scores = {}

    for win_size in range(10, 125, 10):  # 10s, 20s, ..., 60s
        smoothed_preds = apply_3c(
            detection_df[["timestamp"]],
            final_predictions,
            name=name,
            size=win_size
        )

        f1 = f1_score(final_ground_truth, smoothed_preds, average="macro", zero_division=1)
        f1_window_scores[f"{win_size}s"] = round(f1, 4)
        logging.info(f"[{name}] F1 @ {win_size}s = {f1:.4f}")

  
    json_path = os.path.join(output_dir, f"2_stage_f1_timewindow_{name.replace(' ', '_')}.json")
    with open(json_path, "w") as f:
        json.dump(f1_window_scores, f, indent=4)

    logging.info(f"Final 3-class F1 scores across windows saved at: {json_path}")

    