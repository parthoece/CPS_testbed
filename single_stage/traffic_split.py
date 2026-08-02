import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Setup logging
log_file = "traffic_split.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set up at {output_dir}")

def load_and_clean_data(file_path, dataset_name):
    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8", low_memory=False)
    except Exception as e:
        logging.error(f"Error reading {dataset_name}: {e}")
        return pd.DataFrame()
    
    logging.info(f"{dataset_name} initial shape: {df.shape}")
    df = df.dropna(subset=['startDate'])
    selected_features = ['rBytesAvg', 'sBytesAvg', 'sFinRate', 'rFinRate', 'sSynRate', 'rSynRate',
                         'sRstRate', 'rRstRate', 'rttl', 'sttl', 'sAckRate', 'rAckRate',
                         'sAckDelayMax', 'rAckDelayMax', 'sPackets', 'rPackets', 'protocol',
                         'sWinTCP', 'rWinTCP', 'rPayloadAvg', 'sPayloadAvg', 'sInterPacketAvg',
                         'rInterPacketAvg', 'startDate']
    df = df[[col for col in selected_features if col in df.columns]]
    
    if 'protocol' in df.columns:
        df['protocol'] = df['protocol'].astype(str).fillna("unknown")
        df['protocol'] = LabelEncoder().fit_transform(df['protocol'])
    
    df.fillna(0, inplace=True)
    logging.info(f"{dataset_name} cleaned. Final shape: {df.shape}")
    return df

def assign_labels(df, label_value):
    if df.empty:
        return df
    df['label'] = label_value
    return df

def split_data(df, train_ratio, dataset_name):
    train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=42)
    logging.info(f"{dataset_name}: Split into train ({train_df.shape}) and test ({test_df.shape})")
    return train_df, test_df

def split_validation_and_test(test_data, dataset_name):
    val_data, test_data = train_test_split(test_data, test_size=0.80, random_state=42)
    logging.info(f"{dataset_name}: Split into validation ({val_data.shape}) and test ({test_data.shape})")
    return val_data, test_data

def save_labels(df, output_dir, file_prefix):
    labels = df[['label']]
    labels.to_csv(os.path.join(output_dir, f"{file_prefix}_labels.csv"), index=False)
    np.save(os.path.join(output_dir, f"{file_prefix}_labels.npy"), labels.to_numpy())
    logging.info(f"Saved labels for {file_prefix} with shape: {labels.shape}")

def scale_and_save_data(df, output_dir, file_prefix, scaler=None):
    if df.empty:
        logging.warning(f"{file_prefix} dataset is empty. Skipping scaling and saving.")
        return None
    
    # Preserve startDate before dropping it
    if 'startDate' in df.columns:
        df['startDate'] = pd.to_datetime(df['startDate'])  # Keep it in datetime format
        startdate_df = df[['startDate']]
        startdate_df.to_csv(os.path.join(output_dir, f"{file_prefix}_timestamp_traffic.csv"), index=False)
        logging.info(f"Saved startDate timestamps for {file_prefix} with shape: {startdate_df.shape}")
    
    df_numeric = df.drop(columns=['startDate', 'label'], errors='ignore')  # Drop startDate for scaling
    
    if df_numeric.empty:
        logging.warning(f"{file_prefix} has no valid numerical features. Skipping scaling and saving.")
        return None
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df_numeric)
        joblib.dump(scaler, os.path.join(output_dir, f"{file_prefix}_scaler.pkl"))
    
    scaled_data = scaler.transform(df_numeric)
    scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)

    # Save scaled numerical data
    scaled_df.to_csv(os.path.join(output_dir, f"{file_prefix}_scaled_traffic.csv"), index=False)
    np.save(os.path.join(output_dir, f"{file_prefix}_scaled_traffic.npy"), scaled_df.to_numpy())
    
    logging.info(f"Saved scaled data for {file_prefix} with shape: {scaled_df.shape}")
    return scaler

def main_pipeline():
    output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/sl/traffic"
    setup_output_dir(output_dir)
    
    datasets = {
        "normal": ("/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/traffic/normal_flow.csv", 0),
        "fault": ("/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/traffic/fault_flow.csv", 1),
        "attack": ("/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/traffic/attack_flow.csv", 2)
    }
    
    processed_data = {key: assign_labels(load_and_clean_data(file, key), label) for key, (file, label) in datasets.items()}
    normal_train, normal_test = split_data(processed_data["normal"], 0.50, "normal")
    normal_test = normal_test.sample(frac=0.20, random_state=42)   # 20% of the 50% test set
    fault_train, fault_test = split_data(processed_data["fault"], 0.70, "fault")
    attack_train, attack_temp = split_data(processed_data["attack"], 0.70, "attack")
    attack_val, attack_test = split_validation_and_test(attack_temp, "attack")
    normal_val, normal_test = split_validation_and_test(normal_test, "normal")
    fault_val, fault_test = split_validation_and_test(fault_test, "fault")
    
    combined_val = pd.concat([normal_val, fault_val, attack_val], ignore_index=True)
    combined_test = pd.concat([normal_test, fault_test, attack_test], ignore_index=True)
    svm_train = pd.concat([normal_train, fault_train, attack_train], ignore_index=True)
    
    save_labels(combined_val, output_dir, "final_validation")
    save_labels(combined_test, output_dir, "final_test")
    save_labels(svm_train, output_dir, "svm_train")
    
    scalers = {}
    scalers["svm"] = scale_and_save_data(svm_train, output_dir, "svm_train")
    scale_and_save_data(combined_val, output_dir, "final_validation", scaler=scalers["svm"])
    scale_and_save_data(combined_test, output_dir, "final_test", scaler=scalers["svm"])
    
    logging.info(f"Final dataset shapes: SVM Train={svm_train.shape}, Validation={combined_val.shape}, Test={combined_test.shape}")
    logging.info("Traffic data pipeline completed successfully.")
    print("Pipeline execution completed. Check the log file for details.")

if __name__ == "__main__":
    main_pipeline()
