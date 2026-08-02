import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import json
import joblib
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_mixed_timestamps(ts):
    try:
        dt = pd.to_datetime(ts, format="%Y-%m-%dT%H:%M:%S.%f%z", errors="raise")
        return dt.tz_localize(None)  
    except ValueError:
        return pd.to_datetime(ts, format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce")

def load_and_convert_timestamps(sensor_timestamps_path, traffic_timestamps_path, log_timestamps_path):
    sensor_timestamps = pd.read_csv(sensor_timestamps_path, names=["timestamp"])
    traffic_timestamps = pd.read_csv(traffic_timestamps_path, names=["timestamp"])
    log_timestamps = pd.read_csv(log_timestamps_path, names=["timestamp"])

    sensor_timestamps["timestamp"] = sensor_timestamps["timestamp"].apply(parse_mixed_timestamps)
    traffic_timestamps["timestamp"] = traffic_timestamps["timestamp"].apply(parse_mixed_timestamps)
    log_timestamps["timestamp"] = log_timestamps["timestamp"].apply(parse_mixed_timestamps)

    if sensor_timestamps["timestamp"].dtype != "datetime64[ns]" or \
       traffic_timestamps["timestamp"].dtype != "datetime64[ns]" or \
       log_timestamps["timestamp"].dtype != "datetime64[ns]":
        logging.error("Timestamp conversion failed for one or more datasets.")
        raise ValueError("Timestamp conversion error.")

    
    sensor_timestamps.to_csv("converted_sensor_timestamps.csv", index=False)
    traffic_timestamps.to_csv("converted_traffic_timestamps.csv", index=False)
    log_timestamps.to_csv("converted_log_timestamps.csv", index=False)

    logging.info("Converted timestamps successfully saved.")
    
    return sensor_timestamps, traffic_timestamps, log_timestamps


def main():
    sensor_timestamps_path = '/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/ensemble/final_test_timestamps_sensor.csv'
    traffic_timestamps_path = '/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/ensemble/final_test_timestamps_traffic.csv'
    log_timestamps_path = '/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/ensemble/test_timestamps_log.csv'

    try:
        sensor_timestamps, traffic_timestamps, log_timestamps = load_and_convert_timestamps(
            sensor_timestamps_path, traffic_timestamps_path, log_timestamps_path)
        logging.info("Timestamps successfully loaded, converted, and saved.")
    except Exception as e:
        logging.error(f"Error in timestamp processing: {e}")

if __name__ == "__main__":
    main()
