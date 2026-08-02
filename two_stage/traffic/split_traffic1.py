import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging
from sklearn.model_selection import train_test_split
import joblib
log_file = "split_traffic1.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Output directory set up at {output_dir}")

def load_and_clean_data(file_path, dataset_name):
    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8", low_memory=False)
    except Exception as e:
        logging.error(f"Error reading {dataset_name}: {e}")
        return pd.DataFrame()

    initial_shape = df.shape
    logging.info(f"{dataset_name} initial shape: {initial_shape}")
    
    # Convert time format if necessary
    #df['start'] = pd.to_datetime(df['start'], errors='coerce')
    #df['end'] = pd.to_datetime(df['end'], errors='coerce')
    #df = df.dropna(subset=['start', 'end'])  # Ensure no missing timestamps
    df = df.dropna(subset=['startDate']) 
    
    """Initial columns: ['sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate', 'endDate', 'start', 'end', 'startOffset', 
                      'endOffset', 'duration', 'sPackets', 'rPackets', 'sBytesSum', 'rBytesSum', 'sBytesMax', 'rBytesMax', 'sBytesMin', 'rBytesMin', 
                      'sBytesAvg', 'rBytesAvg', 'sLoad', 'rLoad', 'sPayloadSum', 'rPayloadSum', 'sPayloadMax', 'rPayloadMax', 'sPayloadMin', 'rPayloadMin', 
                      'sPayloadAvg', 'rPayloadAvg', 'sInterPacketAvg', 'rInterPacketAvg', 'sttl', 'rttl', 'sAckRate', 'rAckRate', 'sUrgRate', 'rUrgRate', 
                      'sFinRate', 'rFinRate', 'sPshRate', 'rPshRate', 'sSynRate', 'rSynRate', 'sRstRate', 'rRstRate', 'sWinTCP', 'rWinTCP', 'sFragmentRate', 'rFragmentRate', 
                      'sAckDelayMax', 'rAckDelayMax', 'sAckDelayMin', 'rAckDelayMin', 'sAckDelayAvg', 'rAckDelayAvg']"""
    

    selected_features = ['rBytesAvg', 'sBytesAvg', 'sFinRate', 'rFinRate', 'sSynRate', 'rSynRate',
                         'sRstRate', 'rRstRate', 'rttl', 'sttl', 'sAckRate', 'rAckRate',
                         'sAckDelayMax', 'rAckDelayMax', 'sPackets', 'rPackets', 'protocol',
                         'sWinTCP', 'rWinTCP', 'rPayloadAvg', 'sPayloadAvg', 'sInterPacketAvg',
                         'rInterPacketAvg', 'startDate']

   
    df = df[[col for col in selected_features if col in df.columns]]
    
    if 'protocol' in df.columns:
        df['protocol'] = df['protocol'].astype(str).fillna("unknown")
        le = LabelEncoder()
        df['protocol'] = le.fit_transform(df['protocol'])

    #  Drop columns with too many NaNs (e.g., more than 50% missing)
    #threshold = 0.5 * len(df)
    #df = df.dropna(axis=1, thresh=threshold)
        
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

def save_timestamp_and_label(df, output_dir, file_prefix):
    timestamps = df[['startDate']]
    labels = df[['label']]
    
    timestamps.to_csv(os.path.join(output_dir, f"{file_prefix}_timestamps.csv"), index=False)
    labels.to_csv(os.path.join(output_dir, f"{file_prefix}_labels.csv"), index=False)
    logging.info(f"Saved timestamps and labels for {file_prefix}")



def scale_and_save_data(df, output_dir, file_prefix, model_type, scaler=None):
    
    if df.empty:
        logging.warning(f"{file_prefix} dataset is empty. Skipping scaling and saving.")
        return None

    timestamps = df[['startDate']]
    labels = df[['label']]
    
    df_numeric = df.drop(columns=['startDate', 'label'], errors='ignore')

    logging.info(f"\n[{file_prefix}] Data Logging:")
    logging.info(f"  - Original dataset shape: {df.shape}")
    logging.info(f"  - Features before scaling: {df_numeric.shape}")
    logging.info(f"  - Timestamps shape: {timestamps.shape}, Labels shape: {labels.shape}")


    if df_numeric.empty:
        logging.warning(f"{file_prefix} has no valid numerical features. Skipping scaling and saving.")
        return None

    if model_type == "isolation_forest":
        np.save(os.path.join(output_dir, f"{file_prefix}_isolation_forest.npy"), df_numeric.values)
        return
    
    if scaler is None:
        scaler = StandardScaler() if model_type == "oneclass_svm" else MinMaxScaler()
        scaler.fit(df_numeric)
        joblib.dump(scaler, os.path.join(output_dir, f"{file_prefix}_{model_type}_scaler.pkl"))
    
    scaled_data = scaler.transform(df_numeric)

    scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)

    logging.info(f"{file_prefix} - Scaled features shape: {scaled_df.shape}")

    scaled_df.to_csv(os.path.join(output_dir, f"{file_prefix}_{model_type}_scaled.csv"), index=False)
    np.save(os.path.join(output_dir, f"{file_prefix}_{model_type}_scaled.npy"), scaled_df.to_numpy())

    timestamps.to_csv(os.path.join(output_dir, f"{file_prefix}_timestamps.csv"), index=False)
    labels.to_csv(os.path.join(output_dir, f"{file_prefix}_labels.csv"), index=False)
    np.save(os.path.join(output_dir, f"{file_prefix}_labels.npy"), labels.to_numpy())
    
    logging.info(f"\n[{file_prefix}] Data Logging:")
    logging.info(f"  - Scaled features shape: {scaled_df.shape}")

    return scaler 


def save_raw_data(df, output_dir, file_prefix):
    if df.empty:
        logging.warning(f"{file_prefix} dataset is empty. Skipping saving raw CSV.")
        return
    df.to_csv(os.path.join(output_dir, f"{file_prefix}_raw.csv"), index=False)
    logging.info(f"Saved raw data for {file_prefix} before scaling. Shape: {df.shape}")

def main_pipeline():
    output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/traffic/split1"
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
    attack_train, attack_test = split_data(processed_data["attack"], 0.70, "attack")
    logging.info(f"Normal test 20% of the 50% test set shape: {normal_test .shape}")

    normal_val, normal_test = split_validation_and_test(normal_test, "normal")
    fault_val, fault_test = split_validation_and_test(fault_test, "fault")
    attack_val, attack_test = split_validation_and_test(attack_test, "attack")

    stage2train = pd.concat([fault_train, attack_train], ignore_index=True)
    combined_val = pd.concat([normal_val, fault_val, attack_val], ignore_index=True)
    combined_test = pd.concat([normal_test, fault_test, attack_test], ignore_index=True)
    
    save_timestamp_and_label(combined_val, output_dir, "final_validation")
    save_timestamp_and_label(combined_test, output_dir, "final_test")
    save_timestamp_and_label(stage2train, output_dir, "stage2_train")
    
     # Save raw datasets before scaling
    save_raw_data(normal_train, output_dir, "normal_train")
    save_raw_data(fault_train, output_dir, "fault_train")
    save_raw_data(stage2train, output_dir, "stage2train")
    save_raw_data(combined_val, output_dir, "final_validation")
    save_raw_data(combined_test, output_dir, "final_test")

    save_timestamp_and_label(combined_val, output_dir, "final_validation")
    save_timestamp_and_label(combined_test, output_dir, "final_test")
    save_timestamp_and_label(stage2train, output_dir, "stage2_train")

    models = ["oneclass_svm", "isolation_forest", "autoencoder", "lstm"]
    scalers = {}

    # Scale training data and store scalers
    for model_type in models:
        scalers[model_type] = scale_and_save_data(normal_train, output_dir, "normal_train", model_type)
        scale_and_save_data(fault_train, output_dir, "fault_train", model_type, scaler=scalers[model_type])
        #scale_and_save_data(stage2_train, output_dir, "stage2_train", model_type, scaler=scalers[model_type])

    for model_type in models:
        scale_and_save_data(combined_val, output_dir, "final_validation", model_type, scaler=scalers[model_type])
        scale_and_save_data(combined_test, output_dir, "final_test", model_type, scaler=scalers[model_type])
        scale_and_save_data(stage2train, output_dir, "stage2train", model_type, scaler=scalers[model_type])

    logging.info("Traffic data pipeline completed successfully.")
    print("Pipeline execution completed. Check the log file for details.")

if __name__ == "__main__":
    main_pipeline()
 
