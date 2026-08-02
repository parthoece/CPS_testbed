import pandas as pd
import numpy as np
import os
import gc
import logging
import psutil
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import warnings
import ipaddress
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from joblib import dump, load

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("log_split.log")]
)
logger = logging.getLogger(__name__)

DEFAULT_NORMAL_FILE = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/log/normal_combined_log.csv"
DEFAULT_FAULT_FILE = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/log/fault_combined_log.csv"
DEFAULT_ATTACK_FILE = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/log/attack_combined_logs.csv"
DEFAULT_OUTPUT_DIR = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/sl/log/"
DEFAULT_W2V_MODEL_PATH = "word2vec.model"

class ICSDataPipeline:
    def __init__(self, normal_file, fault_file, attack_file, output_dir):
        self.normal_file = normal_file
        self.fault_file = fault_file
        self.attack_file = attack_file
        self.output_dir = output_dir
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        self.normal_data = pd.read_csv(self.normal_file)
        self.fault_data = pd.read_csv(self.fault_file)
        self.attack_data = pd.read_csv(self.attack_file)

        common_columns = list(set(self.attack_data.columns) & set(self.normal_data.columns) & set(self.fault_data.columns))
        self.normal_data = self.normal_data[common_columns]
        self.fault_data = self.fault_data[common_columns]
        self.attack_data = self.attack_data[common_columns]

        logger.info(f"Loaded datasets: Normal={self.normal_data.shape}, Fault={self.fault_data.shape}, Attack={self.attack_data.shape}")
    
    def extract_severity(self, text):
        
        severity_keywords = ['debug', 'info', 'warning', 'error', 'critical', 'fatal']
        for word in severity_keywords:
            if word in text.lower():
                return word.upper()
        return "UNKNOWN"

    def handle_numerical_categorical_fields(self, df, log_type="normal"):
        if df.empty:
            logger.warning(f" Empty dataframe for {log_type} logs. Skipping processing.")
            return df

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        if "Message" in df.columns:
            df["Message"] = df["Message"].apply(lambda x: self.clean_text(x) if pd.notna(x) else "")

        if "ip_address" in df.columns:
            unique_ips = df["ip_address"].nunique()
            
            if unique_ips > 50:  
                df["ip_address"] = df["ip_address"].apply(lambda x: int(ipaddress.IPv4Address(x)) if pd.notnull(x) else 0)
                df["ip_address"] = StandardScaler().fit_transform(df[["ip_address"]])
            else:  
                df = pd.get_dummies(df, columns=["ip_address"])

        categorical_cols = [col for col in df.select_dtypes(include=["object"]).columns if col not in ["Message"]]
        if categorical_cols:
            df[categorical_cols] = df[categorical_cols].apply(lambda col: self.label_encoder.fit_transform(col.astype(str)))

        logger.info(f" Processed {log_type} logs: {df.shape}")
        return df


    def clean_text(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return ""  

        text = text.lower()

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        text = re.sub(r"[^\w\s.:]", "", text)

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words or word.isnumeric()]
        
        return " ".join(words) if words else text  



    def preprocess_text_tfidf(self, train_texts, val_texts=None, test_texts=None):
       
        logger.info(" Applying TF-IDF transformation...")

        train_cleaned = train_texts.apply(lambda x: self.clean_text(x) if pd.notna(x) else "")

        val_cleaned = val_texts.apply(lambda x: self.clean_text(x) if pd.notna(x) else "") if val_texts is not None else None
        test_cleaned = test_texts.apply(lambda x: self.clean_text(x) if pd.notna(x) else "") if test_texts is not None else None
        self.vectorizer.fit(train_cleaned)

        train_tfidf = self.vectorizer.transform(train_cleaned).toarray()
        val_tfidf = self.vectorizer.transform(val_cleaned).toarray() if val_cleaned is not None else None
        test_tfidf = self.vectorizer.transform(test_cleaned).toarray() if test_cleaned is not None else None

        nonzero_columns = np.any(train_tfidf, axis=0)
        train_tfidf = train_tfidf[:, nonzero_columns]

        if val_tfidf is not None:
            val_tfidf = val_tfidf[:, nonzero_columns]
        if test_tfidf is not None:
            test_tfidf = test_tfidf[:, nonzero_columns]

        logger.info(f" TF-IDF Train Shape: {train_tfidf.shape}")
        if val_tfidf is not None:
            logger.info(f" TF-IDF Validation Shape: {val_tfidf.shape}")
        if test_tfidf is not None:
            logger.info(f" TF-IDF Test Shape: {test_tfidf.shape}")

        return train_tfidf, val_tfidf, test_tfidf

    def split_data(self):
        self.normal_data["label"] = 0
        self.fault_data["label"] = 1
        self.attack_data["label"] = 2

        if "Timestamp" in self.normal_data.columns:
            self.normal_data["Timestamp"] = pd.to_datetime(self.normal_data["Timestamp"], errors="coerce")
        if "Timestamp" in self.fault_data.columns:
            self.fault_data["Timestamp"] = pd.to_datetime(self.fault_data["Timestamp"], errors="coerce")
        if "Timestamp" in self.attack_data.columns:
            self.attack_data["Timestamp"] = pd.to_datetime(self.attack_data["Timestamp"], errors="coerce")

        normal_train, normal_test = train_test_split(self.normal_data, train_size=0.50, random_state=42)
        normal_test = normal_test.sample(frac=0.20, random_state=42)
        normal_val, normal_test = train_test_split(normal_test, test_size=0.80, random_state=42)

        fault_train, fault_test = train_test_split(self.fault_data, train_size=0.70, random_state=42)
        fault_val, fault_test = train_test_split(fault_test, test_size=0.80, random_state=42)

        attack_train, attack_temp = train_test_split(self.attack_data, train_size=0.70, random_state=42)
        attack_val, attack_test = train_test_split(attack_temp, test_size=0.67, random_state=42)

        val_data = pd.concat([normal_val, fault_val, attack_val], ignore_index=True)
        test_data = pd.concat([normal_test, fault_test, attack_test], ignore_index=True)
        train_data = pd.concat([normal_train, fault_train, attack_train], ignore_index=True)

        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        logger.info(f" Final Splits: Train={train_data.shape}, Val={val_data.shape}, Test={test_data.shape}")

        return train_data, val_data, test_data
    
    def save_csv_data(self, df, filename):
        file_path = os.path.join(self.output_dir, filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved CSV: {file_path}")
   
    def save_labels_and_Timestamps(self, data, prefix):
        labels = data["label"].values if "label" in data.columns else None
        Timestamps = data["Timestamp"] if "Timestamp" in data.columns else None

        if Timestamps is not None:
            Timestamps = pd.to_datetime(Timestamps, errors='coerce') 
            Timestamps = Timestamps.dropna()  
            Timestamps = Timestamps.dt.strftime('%Y-%m-%d %H:%M:%S') 

        if labels is not None:
            np.save(os.path.join(self.output_dir, f"{prefix}_labels.npy"), labels)
            logger.info(f" Saved {prefix} labels as .npy")
            pd.DataFrame({"label": labels}).to_csv(os.path.join(self.output_dir, f"{prefix}_labels.csv"), index=False)
            logger.info(f" Saved {prefix} labels as CSV")

        if Timestamps is not None and not Timestamps.empty:
            np.save(os.path.join(self.output_dir, f"{prefix}_Timestamps.npy"), Timestamps.values)
            logger.info(f" Saved {prefix} Timestamps as .npy")
            pd.DataFrame({"Timestamp": Timestamps}).to_csv(os.path.join(self.output_dir, f"{prefix}_Timestamps.csv"), index=False)
            logger.info(f" Saved {prefix} Timestamps as CSV")

    def prepare_data(self):
        self.load_data()
        train_data, val_data, test_data = self.split_data()

        train_data = self.handle_numerical_categorical_fields(train_data)
        val_data = self.handle_numerical_categorical_fields(val_data)
        test_data = self.handle_numerical_categorical_fields(test_data)

        self.save_labels_and_Timestamps(self.train_data, "train")
        self.save_labels_and_Timestamps(self.val_data, "val")
        self.save_labels_and_Timestamps(self.test_data, "test")

        X_train, X_val, X_test = self.preprocess_text_tfidf(
            train_data["Message"], val_data["Message"], test_data["Message"]
        )
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        dump(scaler, os.path.join(self.output_dir, "scaler.joblib"))

        np.save(os.path.join(self.output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(self.output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.output_dir, "y_train.npy"), train_data["label"].values)
        np.save(os.path.join(self.output_dir, "y_val.npy"), val_data["label"].values)
        np.save(os.path.join(self.output_dir, "y_test.npy"), test_data["label"].values)

        # Save labels as CSV for easy reference
        train_data[["label"]].to_csv(os.path.join(self.output_dir, "y_train.csv"), index=False)
        val_data[["label"]].to_csv(os.path.join(self.output_dir, "y_val.csv"), index=False)
        test_data[["label"]].to_csv(os.path.join(self.output_dir, "y_test.csv"), index=False)

        logger.info(" Labels saved in both .npy and .csv formats!")


        logger.info(" Data preprocessed and saved for SVM!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICS Data Preprocessing Pipeline")
    parser.add_argument("--normal_file", type=str, default=DEFAULT_NORMAL_FILE)
    parser.add_argument("--fault_file", type=str, default=DEFAULT_FAULT_FILE)
    parser.add_argument("--attack_file", type=str, default=DEFAULT_ATTACK_FILE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w2v_model_path", type=str, default=DEFAULT_W2V_MODEL_PATH)

    args = parser.parse_args()

    pipeline = ICSDataPipeline(
        normal_file=args.normal_file,
        fault_file=args.fault_file,
        attack_file=args.attack_file,
        output_dir=args.output_dir
    )

    pipeline.prepare_data()
