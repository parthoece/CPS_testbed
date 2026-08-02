import os
import numpy as np
import pandas as pd
import logging
import joblib
import json
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Logging Setup ==========
logging.basicConfig(
    filename="ensemble_xgboost.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# ========== Paths ==========
data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/ensemble"
output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/ensemble/xgboost/output"
model_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/Code/single_stage/ensemble/xgboost"
os.makedirs(output_dir, exist_ok=True)

# ========== Load Models ==========
models = {
    "Sensor": joblib.load(os.path.join(model_dir, "xgboost_model_trial_0_sensor.pkl")),
    "Log": joblib.load(os.path.join(model_dir, "xgboost_model_trial_0_log.pkl")),
    "Traffic": joblib.load(os.path.join(model_dir, "xgboost_model_trial_2_traffic.pkl")),
}

# ========== Load Test Data ==========
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
    min_len = min(
        len(test_data[source]["timestamps"]),
        len(test_data[source]["features"]),
        len(test_data[source]["labels"])
    )
    test_data[source]["timestamps"] = test_data[source]["timestamps"].iloc[:min_len]
    test_data[source]["features"] = test_data[source]["features"][:min_len]
    test_data[source]["labels"] = test_data[source]["labels"][:min_len]

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

def ensure_feature_match(aligned_data, model):
    expected_features = model.n_features_in_
    if aligned_data.shape[1] != expected_features:
        if aligned_data.shape[1] > expected_features:
            return aligned_data[:, :expected_features]
        else:
            raise ValueError("Mismatch in feature size and model input.")
    return aligned_data

def window(predictions, window_size=5, source=None, thresholds=None):
    win = predictions.copy()
    half_window = window_size // 2

    if thresholds is None:
        thresholds = {
            "Sensor": {"value": 2, "min_ratio": 0.9},
            "Log": {"value": 1, "min_ratio": 0.4},
            "Traffic": {"value": 0, "min_ratio": 0.9},
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
    for i in range(len(preds)):
        if preds[i] == rule["class"] and probs[i][rule["class"]] < rule["threshold"]:
            preds[i] = 0
    return preds



aligned_data = {}
aligned_labels = {}
for source in ["Sensor", "Log", "Traffic"]:
    features, labels = align_data(all_timestamps, test_data[source]["timestamps"],
                                  test_data[source]["features"], test_data[source]["labels"])
    aligned_data[source] = ensure_feature_match(features, models[source])
    aligned_labels[source] = labels


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
    "Sensor": {"class": 2, "threshold": 0.95},
    "Log": {"class": 1, "threshold": 0.7},
    "Traffic": {"class": 0, "threshold": 0.5}
}


results_dict = {}
for name, sources in combinations.items():
    logging.info(f"Processing combination: {name}")
    source_thresholds = {
        "Sensor": {"value": 2, "min_ratio": 0.9},
        "Log": {"value": 1, "min_ratio": 0.4 if len(sources) > 1 else 0.7},
        "Traffic": {"value": 0, "min_ratio": 0.01}
    }

    predictions = []
    for source in sources:
        if source in ["Sensor", "Traffic"]:
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
    final_predictions[remaining_indices] = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x, minlength=3)),
        axis=0, arr=np.array(predictions)[:, remaining_indices[0]]
    )

    fusion_source = next((src for src in ["Sensor", "Log", "Traffic"] if src in sources), None)
    final_predictions = window(final_predictions, window_size=60, source=fusion_source, thresholds=source_thresholds)

    accuracy = accuracy_score(combination_ground_truth, final_predictions)
    precision = precision_score(combination_ground_truth, final_predictions, average="macro", zero_division=1)
    recall = recall_score(combination_ground_truth, final_predictions, average="macro", zero_division=1)
    f1 = f1_score(combination_ground_truth, final_predictions, average="macro", zero_division=1)
    conf_matrix = confusion_matrix(combination_ground_truth, final_predictions)
    class_report = classification_report(combination_ground_truth, final_predictions, output_dict=True)
    cm_percent = np.nan_to_num(conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True) * 100)

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
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Normal", "Fault", "Attack"],
                yticklabels=["Normal", "Fault", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({name.replace(' ', '')})")
    plt.savefig(cm_filename)
    plt.close()


json_results_path = os.path.join(output_dir, "single_stage.json")
with open(json_results_path, "w") as f:
    json.dump(results_dict, f, indent=4)
logging.info(f"Single-stage detection results saved at {json_results_path}")
