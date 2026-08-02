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
    filename="timewindow_singlestage.log",
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
    "Sensor": joblib.load(os.path.join(model, "best_lgbm_sensor.pkl")),
    "Log": joblib.load(os.path.join(model, "best_lgbm_log.pkl")),
    "Traffic": joblib.load(os.path.join(model, "best_lgbm_traffic.pkl")),
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
def sliding_predict_and_evaluate(model, features_df, label_df, window_size=5, max_time=60):
    results = []
    timestamps = pd.to_datetime(features_df["timestamp"])
    labels_df = label_df.set_index("timestamp")
    start_time = timestamps.min()

    for t in range(0, max_time + 1, window_size):
        window_start = start_time + pd.Timedelta(seconds=t)
        window_end = window_start + pd.Timedelta(seconds=window_size)
        window_mask = (timestamps >= window_start) & (timestamps < window_end)

        if window_mask.sum() == 0:
            results.append({"time": t, "f1_score": None})
            continue

        window_features = features_df[window_mask].drop(columns=["timestamp"]).values
        window_timestamps = features_df[window_mask]["timestamp"]
        window_labels = labels_df.loc[window_timestamps]["label"].values

        preds = model.predict(window_features)
        f1 = f1_score(window_labels, preds, average="macro", zero_division=1)
        results.append({"time": t, "f1_score": f1})

    return results


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
    
    
}

prob_thresholds = {
    "Sensor": {"class": 2, "threshold": 0.67},
    "Log": {"class": 1, "threshold": 0.7},
    "Traffic": {"class": 0, "threshold": 0.5}
}


results_dict = {}
f1_window_scores = {}

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
        source_windowed_preds = {}

        if source == "Sensor" and sources == ["Sensor"]:
            thresholded_preds = apply_prob_threshold(models[source], aligned_data[source], source, prob_thresholds)
            for win_size in range(10, 125, 10):
                windowed = window(thresholded_preds, window_size=win_size, source=source, thresholds=source_thresholds)
                source_windowed_preds[win_size] = windowed
        else:
            raw_preds = models[source].predict(aligned_data[source])
            for win_size in range(10, 125, 10):
                windowed = window(raw_preds, window_size=win_size, source=source, thresholds=source_thresholds)
                source_windowed_preds[win_size] = windowed

        predictions.append(source_windowed_preds) 


  
    combination_ground_truth = np.max([aligned_labels[source] for source in sources], axis=0)
   
    if "Sensor" in sources:
        fusion_source = "Sensor"
    elif "Log" in sources:
        fusion_source = "Log"
    elif "Traffic" in sources:
        fusion_source = "Traffic"
    else:
        fusion_source = None 
 
    window_f1_scores = {}
   
    # Loop through each window size
    for win_size in range(10, 125, 10):  # 10, 20, ..., 60
        logging.info(f"Evaluating with window size: {win_size}s")
        preds_for_win = [source_windowed_preds[win_size] for source_windowed_preds in predictions]
        preds_for_win = np.array(preds_for_win) 

        final_preds_windowed = np.zeros(len(all_timestamps), dtype=int)
        attack_mask = np.any(preds_for_win == 2, axis=0)
        final_preds_windowed[attack_mask] = 2

        remaining_indices = np.where(final_preds_windowed == 0)
        if len(remaining_indices[0]) > 0:
            final_preds_windowed[remaining_indices] = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, minlength=3)), axis=0,
                arr=preds_for_win[:, remaining_indices[0]]
            )

        final_preds_windowed = window(
            final_preds_windowed, window_size=win_size, source=fusion_source, thresholds=source_thresholds
        )

        f1 = f1_score(combination_ground_truth, final_preds_windowed, average="macro", zero_division=1)
        window_f1_scores[f"{win_size}s"] = f1
        logging.info(f"F1 @ {win_size}s = {f1:.4f}")


    f1_out_path = os.path.join(output_dir, f"single_stage_f1_multiple_windows.json")
    with open(f1_out_path, "w") as f:
        json.dump(window_f1_scores, f, indent=4)
    logging.info(f"F1 scores across window sizes saved to: {f1_out_path}")


