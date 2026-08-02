import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    filename="align_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logging.info("Starting the optimized ensemble alignment process...")

data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/ensemble/data/new"

files = {file: os.path.join(data_dir, file) for file in os.listdir(data_dir)}

sensor_timestamps = pd.read_csv(files.get("test_timestamp_sensor.csv"))
log_timestamps = pd.read_csv(files.get("test_timestamp_log.csv"))
traffic_timestamps = pd.read_csv(files.get("test_timestamp_traffic.csv"))

log_timestamps.rename(columns={"formatted_timestamp": "Timestamp"}, inplace=True)
traffic_timestamps.rename(columns={"timestmap": "Timestamp"}, inplace=True)

sensor_timestamps["Timestamp"] = pd.to_datetime(sensor_timestamps["Timestamp"])
log_timestamps["Timestamp"] = pd.to_datetime(log_timestamps["Timestamp"])
traffic_timestamps["Timestamp"] = pd.to_datetime(traffic_timestamps["Timestamp"])

sensor_features = np.load(files.get("test_sensor.npy"))
log_features = np.load(files.get("test_log.npy"))
traffic_features = np.load(files.get("test_traffic.npy"))

sensor_labels = np.load(files.get("test_labels_sensor.npy"))
log_labels = np.load(files.get("test_labels_log.npy"))
traffic_labels = np.load(files.get("test_labels_traffic.npy"))

sensor_timestamps = sensor_timestamps.iloc[:len(sensor_features)]
log_timestamps = log_timestamps.iloc[:len(log_features)]
traffic_timestamps = traffic_timestamps.iloc[:len(traffic_features)]

logging.info(f"Trimmed sensor timestamps to {sensor_timestamps.shape[0]}")
logging.info(f"Trimmed log timestamps to {log_timestamps.shape[0]}")
logging.info(f"Trimmed traffic timestamps to {traffic_timestamps.shape[0]}")


timestamps_list = {
    "sensor": sensor_timestamps,
    "log": log_timestamps,
    "traffic": traffic_timestamps
}
reference_source = max(timestamps_list, key=lambda x: timestamps_list[x].shape[0])
reference_timestamps = timestamps_list[reference_source]

logging.info(f"Using '{reference_source}' timestamps as reference with shape: {reference_timestamps.shape}")

aligned_df = reference_timestamps.copy()
aligned_df = aligned_df.merge(sensor_timestamps, on="Timestamp", how="outer")
aligned_df = aligned_df.merge(log_timestamps, on="Timestamp", how="outer")
aligned_df = aligned_df.merge(traffic_timestamps, on="Timestamp", how="outer")

aligned_df = aligned_df.sort_values(by="Timestamp").reset_index(drop=True)

def align_features(reference_df, timestamps_df, feature_array):
    """Aligns feature array with reference timestamps by trimming any extra rows and filling missing rows with zeros."""
    min_length = min(len(reference_df), len(feature_array))

    feature_array_trimmed = feature_array[:min_length]

    aligned_feature_df = pd.DataFrame(feature_array_trimmed)
    aligned_feature_df["Timestamp"] = timestamps_df["Timestamp"].values[:min_length]  


    aligned_feature_df = reference_df[["Timestamp"]].merge(aligned_feature_df, on="Timestamp", how="left")

    aligned_feature_df.fillna(0, inplace=True)

    return aligned_feature_df.drop(columns=["Timestamp"]).values  

aligned_sensor_features = align_features(aligned_df, sensor_timestamps, sensor_features)
aligned_log_features = align_features(aligned_df, log_timestamps, log_features)
aligned_traffic_features = align_features(aligned_df, traffic_timestamps, traffic_features)


aligned_df["label_sensor"] = aligned_df["Timestamp"].map(dict(zip(sensor_timestamps["Timestamp"], sensor_labels))).fillna(0)
aligned_df["label_log"] = aligned_df["Timestamp"].map(dict(zip(log_timestamps["Timestamp"], log_labels))).fillna(0)
aligned_df["label_traffic"] = aligned_df["Timestamp"].map(dict(zip(traffic_timestamps["Timestamp"], traffic_labels))).fillna(0)


def compute_ground_truth(label_sets):
    return np.where(np.any(label_sets == 2, axis=1), 2,
                    np.where(np.any(label_sets == 1, axis=1), 1, 0))

sensor_traffic_ground_truth = compute_ground_truth(np.column_stack((aligned_df["label_sensor"], aligned_df["label_traffic"])))
sensor_log_ground_truth = compute_ground_truth(np.column_stack((aligned_df["label_sensor"], aligned_df["label_log"])))
traffic_log_ground_truth = compute_ground_truth(np.column_stack((aligned_df["label_traffic"], aligned_df["label_log"])))
sensor_log_traffic_ground_truth = compute_ground_truth(
    np.column_stack((aligned_df["label_sensor"], aligned_df["label_log"], aligned_df["label_traffic"]))
)

np.save(os.path.join(data_dir, "aligned_sensor_traffic_features.npy"), np.column_stack((aligned_sensor_features, aligned_traffic_features)))
np.save(os.path.join(data_dir, "aligned_sensor_log_features.npy"), np.column_stack((aligned_sensor_features, aligned_log_features)))
np.save(os.path.join(data_dir, "aligned_traffic_log_features.npy"), np.column_stack((aligned_traffic_features, aligned_log_features)))
np.save(os.path.join(data_dir, "aligned_sensor_log_traffic_features.npy"), np.column_stack((aligned_sensor_features, aligned_log_features, aligned_traffic_features)))

np.save(os.path.join(data_dir, "aligned_sensor_traffic_ground_truth.npy"), sensor_traffic_ground_truth)
np.save(os.path.join(data_dir, "aligned_sensor_log_ground_truth.npy"), sensor_log_ground_truth)
np.save(os.path.join(data_dir, "aligned_traffic_log_ground_truth.npy"), traffic_log_ground_truth)
np.save(os.path.join(data_dir, "aligned_sensor_log_traffic_ground_truth.npy"), sensor_log_traffic_ground_truth)

aligned_df.to_csv(os.path.join(data_dir, "aligned_sensor_traffic_labels.csv"), index=False)
aligned_df.to_csv(os.path.join(data_dir, "aligned_sensor_log_labels.csv"), index=False)
aligned_df.to_csv(os.path.join(data_dir, "aligned_traffic_log_labels.csv"), index=False)
aligned_df.to_csv(os.path.join(data_dir, "aligned_sensor_log_traffic_labels.csv"), index=False)


if (
    sensor_traffic_ground_truth.shape[0] == aligned_sensor_features.shape[0] and
    sensor_log_ground_truth.shape[0] == aligned_sensor_features.shape[0] and
    traffic_log_ground_truth.shape[0] == aligned_traffic_features.shape[0] and
    sensor_log_traffic_ground_truth.shape[0] == aligned_sensor_features.shape[0]
):
    logging.info(" Features & Labels aligned correctly for all combinations.")
else:
    logging.error(" Mismatch detected! Check feature-label alignment.")
    
logging.info(" Features & Labels aligned correctly by trimming extra timestamps.")
logging.info(f"Final reference timestamps: {reference_source}, total rows: {aligned_df.shape[0]}")
logging.info("Alignment process completed successfully!")
