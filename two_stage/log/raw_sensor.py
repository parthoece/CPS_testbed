import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Setup logging
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "raw_sensor.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Sensor Preprocessing Initialized")

def detect_and_rename_timestamp_column(df):
    time_keywords = ['time', 'timestamp', 'date']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            logging.info(f"Detected time-related column '{col}', renaming it to 'timestamp'.")
            df = df.rename(columns={col: 'timestamp'})
            return df
    return df

def clean_and_prepare_plc_data(df, plc_name):
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^\s+$')]
    df = df.dropna(axis=1, how='all')
    df = detect_and_rename_timestamp_column(df)

    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure timestamp is parsed as datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    logging.info(f"Cleaned {plc_name} data shape: {df.shape}")
    return df



def merge_plc_data(plc1, plc2):
    
    if 'timestamp' in plc1.columns:
        plc1['timestamp'] = pd.to_datetime(plc1['timestamp'], errors='coerce')
    if 'timestamp' in plc2.columns:
        plc2['timestamp'] = pd.to_datetime(plc2['timestamp'], errors='coerce')

    matching_columns = list(set(plc1.columns) & set(plc2.columns) - {'timestamp'})
    if matching_columns:
        logging.info(f"Dropping matching columns from PLC2: {matching_columns}")
        plc2 = plc2.drop(columns=matching_columns)

    plc1 = plc1.sort_values(by='timestamp')
    plc2 = plc2.sort_values(by='timestamp')

    merged = pd.merge_asof(plc1, plc2, on='timestamp', direction='nearest')
    logging.info(f"Merged PLC data shape: {merged.shape}")
    return merged


def preprocess_and_save(df, output_dir, drop_columns, model_type, label_value, dataset_name):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^\s+$')]

    timestamp = df['timestamp'] if 'timestamp' in df.columns else pd.Series([None] * len(df))
    df = df.drop(columns=['timestamp'], errors='ignore')

    df['label'] = label_value

    cleaned_file = os.path.join(output_dir, f"{dataset_name}_cleaned_data_with_label.csv")
    cleaned_df = pd.concat([timestamp, df], axis=1) if not timestamp.isnull().all() else df
    cleaned_df.to_csv(cleaned_file, index=False)
    logging.info(f"Cleaned data for {dataset_name} saved to {cleaned_file}")



    labels = df['label']
    df = df.drop(columns=['label'])

    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    if model_type in ["oneclass_svm", "isolation_forest"]:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
    elif model_type in ["autoencoder", "lstm"]:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    np.save(os.path.join(output_dir, f"{dataset_name}_{model_type}_scaled_data.npy"), scaled_data)
    np.save(os.path.join(output_dir, f"{dataset_name}_labels.npy"), labels)
    return scaled_data

if __name__ == "__main__":
    output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/sensor/"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    datasets_paths = {
        "normal": (
            '/home/smasher/Downloads/cps_tesbed/data/normal/sensor/normal_snapshots_PLC1.csv',
            '/home/smasher/Downloads/cps_tesbed/data/normal/sensor/normal_snapshots_PLC2.csv'
        ),
        "fault": (
            '/home/smasher/Downloads/cps_tesbed/data/faults/sensor/faults_snapshots_PLC1.csv',
            '/home/smasher/Downloads/cps_tesbed/data/faults/sensor/faults_snapshots_PLC2.csv'
        ),
        "attack": (
            '/home/smasher/Downloads/cps_tesbed/data/attack_data/attack_snapshots_PLC1.csv',
            '/home/smasher/Downloads/cps_tesbed/data/attack_data/attack_snapshots_PLC2.csv'
        )
    }

    labels = {
        "normal": 0,
        "fault": 1,
        "attack": 2
    }

    #columns_to_drop = ['current_loop', 'loop_latency', 'logic_execution_time', ' ']
    columns_to_drop = ['logic_execution_time', ' ']

    model_types = ["oneclass_svm", "isolation_forest", "autoencoder", "lstm"]

    for dataset_name, (plc1_path, plc2_path) in datasets_paths.items():
        plc1_data = clean_and_prepare_plc_data(pd.read_csv(plc1_path), f"{dataset_name}_PLC1")
        plc2_data = clean_and_prepare_plc_data(pd.read_csv(plc2_path), f"{dataset_name}_PLC2")

        merged_data = merge_plc_data(plc1_data, plc2_data)

        for model in model_types:
            preprocess_and_save(merged_data, output_dir, columns_to_drop, model, labels[dataset_name], dataset_name)

    print("Preprocessing and merging completed successfully.")
