import os
import numpy as np
import pandas as pd
import logging
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

logging.basicConfig(
    filename="ensemble_lgbm.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/ensemble"
model = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/ensemble/lgbm"
output_dir = os.path.join(model, "output/")
os.makedirs(output_dir, exist_ok=True)

models = {
    "Sensor": joblib.load(os.path.join(data_dir, "best_lgbm_sensor.pkl")),
    "Log": joblib.load(os.path.join(data_dir, "best_lgbm_log.pkl")),
    "Traffic": joblib.load(os.path.join(data_dir, "best_lgbm_traffic.pkl")),
}

test_data = {
    "Sensor": {
        "timestamps": pd.read_csv(os.path.join(data_dir, "test_timestamp_sensor.csv")),
        "features": np.load(os.path.join(data_dir, "test_scaled_sensor.npy")),
        "labels": np.load(os.path.join(data_dir, "test_labels_sensor.npy")),
    },
    "Log": {
        "timestamps": pd.read_csv(os.path.join(data_dir, "test_timestamp_log.csv")),
        "features": np.load(os.path.join(data_dir, "test_scaled_log.npy")),
        "labels": np.load(os.path.join(data_dir, "test_labels_log.npy")),
    },
    "Traffic": {
        "timestamps": pd.read_csv(os.path.join(data_dir, "test_timestamp_traffic.csv")),
        "features": np.load(os.path.join(data_dir, "test_scaled_traffic.npy")),
        "labels": np.load(os.path.join(data_dir, "test_labels_traffic.npy")),
    },
}


for source in ["Sensor", "Log", "Traffic"]:
    test_data[source]["timestamps"]["timestamp"] = pd.to_datetime(test_data[source]["timestamps"]["timestamp"])

for source in ["Sensor", "Log", "Traffic"]:
    num_features = len(test_data[source]["features"])
    num_timestamps = len(test_data[source]["timestamps"])

    if num_timestamps > num_features:
        test_data[source]["timestamps"] = test_data[source]["timestamps"].iloc[:num_features]
    elif num_features > num_timestamps:
        test_data[source]["features"] = test_data[source]["features"][:num_timestamps]


all_timestamps = pd.concat([
    test_data["Sensor"]["timestamps"], 
    test_data["Log"]["timestamps"], 
    test_data["Traffic"]["timestamps"]
]).drop_duplicates().sort_values("timestamp").reset_index(drop=True)


def align_data(reference_timestamps, timestamps_df, data, labels):
    timestamps_df = timestamps_df.sort_values("timestamp").reset_index(drop=True)
    labels = labels.flatten() if labels.ndim > 1 else labels
    min_length = min(len(timestamps_df), len(data), len(labels))
    timestamps_df = timestamps_df.iloc[:min_length]
    data = data[:min_length]
    labels = labels[:min_length]

    df_features = pd.DataFrame({"timestamp": timestamps_df["timestamp"]})
    for i in range(data.shape[1]):  
        df_features[f"feature_{i}"] = data[:, i]

    df_labels = pd.DataFrame({"timestamp": timestamps_df["timestamp"], "label": labels})

    aligned_features = pd.merge_asof(
        reference_timestamps, df_features, on="timestamp",
        tolerance=pd.Timedelta(seconds=60), direction="nearest"
    ).fillna(0)

    aligned_labels = pd.merge_asof(
        reference_timestamps, df_labels, on="timestamp",
        tolerance=pd.Timedelta(seconds=60), direction="nearest"
    ).fillna(0)

    return aligned_features.drop(columns=["timestamp"]).values, aligned_labels["label"].values.astype(int)

def ensure_feature_match(aligned_data, model, source_name):
    expected_features = model.n_features_in_  
    actual_features = aligned_data.shape[1] 

    if actual_features != expected_features:
        if actual_features > expected_features:
            aligned_data = aligned_data[:, :expected_features]
        elif actual_features < expected_features:
            raise ValueError

    return aligned_data

def window(predictions, window_size=5, source=None, thresholds=None):
    win = predictions.copy()
    half_window = window_size // 2

    if thresholds is None:
        thresholds = {
            "Sensor": {"value": 2, "min_ratio": 0.5},    
            "Log": {"value": 1, "min_ratio": 0.4},       
            "Traffic": {"value": 0, "min_ratio": 0.5},   
        }

    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        window_vals = predictions[start:end]
        window_len = len(window_vals)

        if source in thresholds:
            rule = thresholds[source]
            label_count = np.count_nonzero(window_vals == rule["value"])
            ratio = label_count / window_len
            if ratio >= rule["min_ratio"]:
                win[i] = rule["value"]
            else:
                win[i] = Counter(window_vals).most_common(1)[0][0]
        else:
            win[i] = Counter(window_vals).most_common(1)[0][0]

    return win
def apply_prob_threshold(model, data, source, thresholds):
    probs = model.predict_proba(data)
    preds = np.argmax(probs, axis=1)

    if source not in thresholds:
        return preds

    rule = thresholds[source]
    target_class = rule["class"]
    min_prob = rule["threshold"]

    for i in range(len(preds)):
        if preds[i] == target_class and probs[i][target_class] < min_prob:
            preds[i] = 0  
    return preds

aligned_data = {}
aligned_labels = {}

for source in ["Sensor", "Log", "Traffic"]:
    aligned_data[source], aligned_labels[source] = align_data(
        all_timestamps, 
        test_data[source]["timestamps"], 
        test_data[source]["features"], 
        test_data[source]["labels"]
    )

for source in ["Sensor", "Log", "Traffic"]:
    aligned_data[source] = ensure_feature_match(aligned_data[source], models[source], source)


combinations = {
    "Sensor + Log + Traffic": ["Sensor", "Log", "Traffic"],
    "Sensor + Traffic": ["Sensor", "Traffic"],
    "Sensor + Log": ["Sensor", "Log"],
    "Log + Traffic": ["Log", "Traffic"],
    "Sensor": ["Sensor"],
    "Log": ["Log"],
    "Traffic": ["Traffic"],
    
}

prob_thresholds = {
    "Sensor": {"class": 2, "threshold": 0.67},
    "Log": {"class": 1, "threshold": 0.7},
    "Traffic": {"class": 0, "threshold": 0.5}
}


results_dict = {}
for name, sources in combinations.items():
    logging.info(f" Processing combination: {name}")
    if len(sources) == 1:
        source_thresholds = {
            "Sensor": {"value": 2, "min_ratio": 0.3},
            "Log": {"value": 1, "min_ratio": 0.7},
            "Traffic": {"value": 0, "min_ratio": 0.2},
        }
    else:
        source_thresholds = {
            "Sensor": {"value": 2, "min_ratio": 0.3},
            "Log": {"value": 1, "min_ratio": 0.4},
            "Traffic": {"value": 0, "min_ratio": 0.5},
        }

    predictions = []
    for source in sources:
        if source == "Sensor" and sources == ["Sensor"]:
            thresholded_preds = apply_prob_threshold(models[source], aligned_data[source], source, prob_thresholds)
            smoothed_preds = window(thresholded_preds, window_size=60, source=source, thresholds=source_thresholds)
        else:
            raw_preds = models[source].predict(aligned_data[source])
            smoothed_preds = window(raw_preds, window_size=60, source=source, thresholds=source_thresholds)
        
        predictions.append(smoothed_preds)

  
    combination_ground_truth = np.max([aligned_labels[source] for source in sources], axis=0)
    final_predictions = np.zeros(len(all_timestamps), dtype=int)  
    attack_mask = np.any(np.array(predictions) == 2, axis=0)  
    final_predictions[attack_mask] = 2  
    remaining_indices = np.where(final_predictions == 0)
    if len(remaining_indices[0]) > 0:
        final_predictions[remaining_indices] = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=3)), axis=0,
            arr=np.array(predictions)[:, remaining_indices[0]]
        )
    
    if "Sensor" in sources:
        fusion_source = "Sensor"
    elif "Log" in sources:
        fusion_source = "Log"
    elif "Traffic" in sources:
        fusion_source = "Traffic"
    else:
        fusion_source = None 
    final_predictions = window(final_predictions, window_size=60, source=fusion_source, thresholds=source_thresholds)
    accuracy = accuracy_score(combination_ground_truth, final_predictions)
    precision = precision_score(combination_ground_truth, final_predictions, average="macro", zero_division=1)
    recall = recall_score(combination_ground_truth, final_predictions, average="macro", zero_division=1)
    f1 = f1_score(combination_ground_truth, final_predictions, average="macro", zero_division=1)
    conf_matrix = confusion_matrix(combination_ground_truth, final_predictions)
    class_report = classification_report(combination_ground_truth, final_predictions, output_dict=True)
    cm_percent = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True) * 100
    cm_percent = np.nan_to_num(cm_percent)  

    logging.info(f" {name} - Final 3-Class Accuracy: {accuracy:.4f}")
    logging.info(f" {name} - Final 3-Class Precision: {precision:.4f}")
    logging.info(f" {name} - Final 3-Class Recall: {recall:.4f}")
    logging.info(f" {name} - Final 3-Class F1 Score: {f1:.4f}")
    logging.info(f" {name} - Final 3-Class Confusion Matrix: /n{conf_matrix}")
    logging.info(f" {name} - Final 3-Class Confusion Matrix (Percent): /n{cm_percent}")

   
    results_dict[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix (Counts)": conf_matrix.tolist(),
        "Confusion Matrix (Percentage)": cm_percent.tolist(),
        "Classification Report": class_report
    }

    cm_filename = os.path.join(output_dir, "cm", f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    os.makedirs(os.path.dirname(cm_filename), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Normal", "Fault", "Attack"], yticklabels=["Normal", "Fault", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix({name.replace(' ', '')})")
    plt.savefig(cm_filename)
    plt.close()
    json_results_path = os.path.join(output_dir, "single_stage.json")
    with open(json_results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    logging.info(f"Single-stage detection results saved successfully at {json_results_path}!")
