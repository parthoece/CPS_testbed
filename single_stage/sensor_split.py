import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

log_file = "sensor_split.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f" starting sensor SL processing")

def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f" Output directory set up at {output_dir}")

def load_and_clean_data(file_path, dataset_name):
    """Load, clean, and sort data by timestamp."""
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    logging.info(f" Columns after stripping spaces in {dataset_name}: {df.columns.tolist()}")

    # Drop unwanted columns
    #columns_to_drop = ['current_loop', 'loop_latency', 'logic_execution_time']
    columns_to_drop = ['current_loop', 'loop_latency']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    nan_count = df.isna().sum().sum()
    if df.isna().any().any():  # Check if any NaN exists in the DataFrame

        logging.warning(f"⚠ {dataset_name} contains {nan_count} NaN values. Filling with 0.")
        df.fillna(0, inplace=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.sort_values(by="timestamp").reset_index(drop=True)  # Sort by time
        logging.info(f" {dataset_name} sorted by timestamp.")
    else:
        logging.warning(f"⚠ No 'timestamp' column found in {dataset_name}! Make sure the dataset is formatted correctly.")

    logging.info(f" {dataset_name} cleaned. Shape after cleaning: {df.shape}")
    return df

def split_data(df, split_ratio, dataset_name):
    train_data, test_data = train_test_split(df, train_size=split_ratio, random_state=42)
    logging.info(f" {dataset_name} train data shape: {train_data.shape}, {dataset_name} test data shape: {test_data.shape}")
    return train_data, test_data

def split_validation_and_test(test_data, dataset_name):
    val_data, final_test_data = train_test_split(test_data, test_size=0.80, random_state=42)
    logging.info(f" {dataset_name} validation shape: {val_data.shape}, {dataset_name} final test shape: {final_test_data.shape}")
    return val_data, final_test_data

def save_timestamp_and_label(df, output_dir, file_prefix, scale_labels=True):
    if 'timestamp' in df.columns and 'label' in df.columns:
        timestamp_and_label = df[['timestamp', 'label']]
        file_path = os.path.join(output_dir, f"{file_prefix}_timestamps_labels.csv")
        timestamp_and_label.to_csv(file_path, index=False)
        logging.info(f" Saved timestamps and labels to {file_path}")
    else:
        logging.warning(f"⚠ No timestamp or label columns found in {df.columns.tolist()}!")

def scale_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    logging.info(" Data scaled using Min-Max scaling.")
    return scaled_data, scaler

def calculate_optimal_timesteps_with_timestamp(timestamp_series):

    if timestamp_series is None or timestamp_series.isna().sum() > 0:
        logging.warning(f"⚠ Invalid or missing timestamps. Defaulting to TIMESTEPS=10.")
        return 10

    timestamp_series = pd.to_datetime(timestamp_series, errors='coerce')

    time_diffs = timestamp_series.diff().dropna().dt.total_seconds()

    logging.info(f"Time differences (in seconds): {time_diffs.head()}")

    avg_time_gap = time_diffs.mean() 
    logging.info(f"Average time gap: {avg_time_gap:.2f} seconds")

    if avg_time_gap == 0 or pd.isna(avg_time_gap):
        logging.warning("⚠ Detected zero or NaN time gaps. Using TIMESTEPS=10")
        return 10 

    best_timestep = max(2, min(int(60 / avg_time_gap), 50)) 

    logging.info(f" Avg Time Gap: {avg_time_gap:.2f}s, Optimal LSTM TIMESTEPS: {best_timestep}")

    return best_timestep


def reshape_for_lstm(data, output_dir, file_prefix, optimal_timesteps):

    if not isinstance(data, np.ndarray):
        logging.error("⚠ Input to reshape_for_lstm must be a NumPy array.")
        raise ValueError("Input to reshape_for_lstm must be a NumPy array.")

    logging.info(f"DataFrame shape before reshaping: {data.shape}")

    num_samples, num_features = data.shape
    logging.info(f" Number of samples: {num_samples}, Number of features: {num_features}")

    num_trim = num_samples % optimal_timesteps
    if num_trim > 0:
        logging.info(f"⚠ Trimming {num_trim} samples to fit optimal TIMESTEPS={optimal_timesteps}")
        data = data[:-num_trim]

    num_samples = data.shape[0] // optimal_timesteps
    reshaped_data = data.reshape(num_samples, optimal_timesteps, num_features)

    logging.info(f" Reshaped data shape: {reshaped_data.shape}")

    reshaped_file = os.path.join(output_dir, f"{file_prefix}_lstm_reshaped.npy")
    np.save(reshaped_file, reshaped_data)
    logging.info(f" Saved LSTM reshaped data for {file_prefix} with shape: {reshaped_data.shape}")

    return reshaped_data


def main_pipeline():
    output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/sl/sensor"
    setup_output_dir(output_dir)

    datasets = {
        "normal": "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/sensor/normal_cleaned_data_with_label.csv",
        "fault": "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/sensor/fault_cleaned_data_with_label.csv",
        "attack": "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/sensor/attack_cleaned_data_with_label.csv"
    }

    normal_data = load_and_clean_data(datasets["normal"], "normal")
    fault_data = load_and_clean_data(datasets["fault"], "fault")
    attack_data = load_and_clean_data(datasets["attack"], "attack")

    logging.info(f" Final dataset shapes: Normal={normal_data.shape}, Fault={fault_data.shape}, Attack={attack_data.shape}")

    normal_train, normal_test = split_data(normal_data, 0.50, "normal")
    normal_test = normal_test.sample(frac=0.20, random_state=42)   # 20% of the 50% test set

    fault_train, fault_test = split_data(fault_data, 0.70, "fault")

    attack_train, attack_temp = split_data(attack_data, 0.70, "attack")
    attack_val, attack_test = split_validation_and_test(attack_temp, "attack")

    normal_val, normal_test = split_validation_and_test(normal_test, "normal")
    fault_val, fault_test = split_validation_and_test(fault_test, "fault")

    combined_val = pd.concat([normal_val, fault_val, attack_val], ignore_index=True)
    combined_test = pd.concat([normal_test, fault_test, attack_test], ignore_index=True)
    svm_train = pd.concat([normal_train, fault_train, attack_train], ignore_index=True)

    logging.info(f"Final combined validation data shape: {combined_val.shape}")
    logging.info(f"Final combined test data shape: {combined_test.shape}")
    logging.info(f"Final SVM train data shape: {svm_train.shape}")

    save_timestamp_and_label(combined_val, output_dir, "final_validation", scale_labels=False)
    save_timestamp_and_label(combined_test, output_dir, "final_test", scale_labels=False)

    svm_train_labels = svm_train["label"].values
    combined_val_labels = combined_val["label"].values
    combined_test_labels = combined_test["label"].values

    svm_train_final = svm_train.drop(columns=['timestamp', 'label'], errors='ignore')
    combined_val_final = combined_val.drop(columns=['timestamp', 'label'], errors='ignore')
    combined_test_final = combined_test.drop(columns=['timestamp', 'label'], errors='ignore')

    svm_train.to_csv(os.path.join(output_dir, "svm_train.csv"), index=False)
    combined_val.to_csv(os.path.join(output_dir, "combined_validation.csv"), index=False)
    combined_test.to_csv(os.path.join(output_dir, "combined_test.csv"), index=False)

    svm_train_scaled, svm_train_scaler = scale_data(svm_train_final)
    combined_val_scaled, _ = scale_data(combined_val_final)
    combined_test_scaled, _ = scale_data(combined_test_final)

    np.save(os.path.join(output_dir, "svm_train_scaled.npy"), np.hstack((svm_train_scaled, svm_train_labels.reshape(-1, 1))))
    np.save(os.path.join(output_dir, "combined_val_scaled.npy"), np.hstack((combined_val_scaled, combined_val_labels.reshape(-1, 1))))
    np.save(os.path.join(output_dir, "combined_test_scaled.npy"), np.hstack((combined_test_scaled, combined_test_labels.reshape(-1, 1))))

    np.save(os.path.join(output_dir, "svm_train_labels.npy"), svm_train_labels)
    np.save(os.path.join(output_dir, "combined_val_labels.npy"), combined_val_labels)
    np.save(os.path.join(output_dir, "combined_test_labels.npy"), combined_test_labels)

    import joblib
    joblib.dump(svm_train_scaler, os.path.join(output_dir, "svm_train_scaler.pkl"))
    
    logging.info(f" SVM training data shape (features + labels): {svm_train_scaled.shape}")
    logging.info(f" Validation data shape (features + labels): {combined_val_scaled.shape}")
    logging.info(f" Test data shape (features + labels): {combined_test_scaled.shape}")
    logging.info(f" SVM training labels shape: {svm_train_labels.shape}")
    logging.info(f" Validation labels shape: {combined_val_labels.shape}")
    logging.info(f" Test labels shape: {combined_test_labels.shape}")
    
    logging.info(" All datasets processed, scaled, and saved.")

if __name__ == "__main__":
    try:
        main_pipeline()
    except Exception as e:
        logging.error(f"⚠ An error occurred: {str(e)}")
