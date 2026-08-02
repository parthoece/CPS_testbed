import os
import numpy as np
import pandas as pd
import logging

# Configure logging
log_file_path = "dataset_shape_correction.log"
logging.basicConfig(
    filename=log_file_path,
    filemode="a",  
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

data_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/ensemble"

expected_filenames = {
    "Sensor": {
        "data": "final_test_oneclass_svm_scaled_data_sensor.npy",
        "timestamps": "final_test_timestamps_sensor.csv",
        "labels": "final_test_labels_sensor.npy",
    },
    "Log": {
        "data": "test_log.npy",
        "timestamps": "test_timestamps_log.csv",
        "labels": "test_labels_log.npy",
    },
    "Traffic": {
        "data": "final_test_oneclass_svm_scaled_traffic.npy",
        "timestamps": "final_test_timestamps_traffic.csv",
        "labels": "final_test_labels_traffic.npy",
    },
}

def process_and_correct_shapes(directory, expected_filenames):
    try:
        filepaths = {key: {} for key in expected_filenames.keys()}

        for key, filenames in expected_filenames.items():
            for data_type, filename in filenames.items():
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    filepaths[key][data_type] = filepath
                else:
                    logging.warning(f"Missing {data_type} for {key}: {filepath}")

        for key, paths in filepaths.items():
            logging.info(f"Processing {key} data:")

            try:
                data = np.load(paths["data"])
                timestamps = pd.read_csv(paths["timestamps"]).values.flatten()
                labels = np.load(paths["labels"]).flatten()

                logging.info(
                    f"{key} Shapes - Data: {data.shape}, Timestamps: {timestamps.shape}, Labels: {labels.shape}"
                )

                min_length = min(len(data), len(timestamps), len(labels))

                data = data[:min_length]
                timestamps = timestamps[:min_length]
                labels = labels[:min_length]

                np.save(paths["data"], data)
                pd.DataFrame(timestamps).to_csv(paths["timestamps"], index=False, header=False)
                np.save(paths["labels"], labels)

                logging.info(
                    f" Fixed Shapes for {key} - Data: {data.shape}, Timestamps: {timestamps.shape}, Labels: {labels.shape}"
                )

            except Exception as e:
                logging.error(f" Error processing {key}: {e}")

            logging.info("-" * 40)

    except Exception as e:
        logging.error(f" Error accessing directory {directory}: {e}")

process_and_correct_shapes(data_dir, expected_filenames)

print(" Data correction completed. Check 'dataset_shape_correction.log' for details.")
