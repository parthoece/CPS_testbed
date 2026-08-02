import os
import numpy as np
import pandas as pd
import logging

log_file_path = 'check_shape.log'
logging.basicConfig(filename=log_file_path,
                    filemode='a',  # Append mode
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/ensemble/data'

expected_filenames = {
    "Sensor": {
        "data": "final_test_oneclass_svm_scaled_data_sensor.npy",
        "timestamps": "final_test_timestamps_sensor.csv",
        "labels": "final_test_labels_sensor.npy"
    },
    "Log": {
        "data": "test_log.npy",
        "timestamps": "test_timestamps_log.csv",
        "labels": "test_labels_log.npy"
    },
    "Traffic": {
        "data": "final_test_oneclass_svm_scaled_traffic.npy",
        "timestamps": "final_test_timestamps_traffic.csv",
        "labels": "final_test_labels_traffic.npy"
    }
}

def process_files_in_dir(directory, expected_filenames):
    try:
        files = os.listdir(directory)
        logging.info(f"Found {len(files)} files in directory: {directory}")

        filepaths = {}
        for key in expected_filenames.keys():
            filepaths[key] = {}

        for file in files:
            for key, filenames in expected_filenames.items():
                for data_type, filename in filenames.items():
                    if file == filename:
                        filepath = os.path.join(directory, file)
                        filepaths[key][data_type] = filepath
                        logging.info(f"Matched {key} {data_type}: {filepath}")
                        break  # Break inner loop if matched

        for key, paths in filepaths.items():
            logging.info(f"Processing {key} data:")

            try:
                data = np.load(paths['data'])
                logging.info(f"{key} Data Shape: {data.shape}")
            except Exception as e:
                logging.error(f"Error loading {key} data: {e}")

            try:
                timestamps = pd.read_csv(paths['timestamps'])
                logging.info(f"{key} Timestamps Shape: {timestamps.shape}")
            except Exception as e:
                logging.error(f"Error loading {key} timestamps: {e}")

            try:
                labels = np.load(paths['labels'])
                logging.info(f"{key} Labels Shape: {labels.shape}")
            except Exception as e:
                logging.error(f"Error loading {key} labels: {e}")

            logging.info("-" * 40)

    except Exception as e:
        logging.error(f"Error accessing directory {directory}: {e}")

process_files_in_dir(data_dir, expected_filenames)
