import pandas as pd
import numpy as np
import os
import logging
import argparse
import gensim
print(gensim.__version__)
import warnings
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import spacy
import re

nlp = spacy.load("en_core_web_sm")  # Load SpaCy for lemmatization

# Ensure NLTK dependencies are downloaded
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./split_lstm_log.log")]
)
logger = logging.getLogger(__name__)

class ICSDataPipeline:
    def __init__(self, normal_file, fault_file, attack_file, output_dir, w2v_model_path):
        self.normal_file = normal_file
        self.fault_file = fault_file
        self.attack_file = attack_file
        self.output_dir = output_dir
        self.w2v_model_path = os.path.join(output_dir, w2v_model_path)  

        os.makedirs(output_dir, exist_ok=True)  

        self.ensure_word2vec_model()

    def ensure_word2vec_model(self):
        if os.path.exists(self.w2v_model_path):
            logger.info(f" Using existing Word2Vec model: {self.w2v_model_path}")
            try:
                self.w2v_model = Word2Vec.load(self.w2v_model_path)
                logger.info(" Loaded full Word2Vec model successfully.")
                return
            except:
                logger.warning("Full model not found, trying KeyedVectors...")

            try:
                from gensim.models import KeyedVectors
                self.w2v_model = KeyedVectors.load(self.w2v_model_path)
                logger.info(" Loaded Word2Vec KeyedVectors model successfully.")
                return
            except Exception as e:
                logger.error(f" Failed to load Word2Vec: {e}, retraining now.")

        self.load_data()

        corpus = (
            self.normal_data["Message"].tolist()
            + self.fault_data["Message"].tolist()
            + self.attack_data["Message"].tolist()
        )

        logger.warning("No valid Word2Vec model found. Training from scratch...")
        self.train_word2vec(corpus)  



    def get_or_train_word2vec(self):
        try:
            import gensim.downloader as api
            logger.info("📥 Downloading Google's pre-trained Word2Vec model...")
            self.w2v_model = api.load("word2vec-google-news-300") 
            self.w2v_model.save(self.w2v_model_path)
            logger.info(f" Downloaded and saved Word2Vec model at {self.w2v_model_path}")

        except Exception as e:
            logger.error(f" Download failed! Training Word2Vec from scratch. Error: {e}")
            self.train_word2vec()


        

    def load_data(self):
        try:
            self.normal_data = pd.read_csv(self.normal_file)
            self.fault_data = pd.read_csv(self.fault_file)
            self.attack_data = pd.read_csv(self.attack_file)

            required_columns = ["Message"]
            for df_name, df in zip(["Normal", "Fault", "Attack"], [self.normal_data, self.fault_data, self.attack_data]):
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f" {df_name} dataset is missing required columns: {missing_cols}")

            common_columns = list(set(self.attack_data.columns) & set(self.normal_data.columns) & set(self.fault_data.columns))
            self.normal_data = self.normal_data[common_columns]
            self.fault_data = self.fault_data[common_columns]
            self.attack_data = self.attack_data[common_columns]

            logger.info(f" Loaded datasets: Normal={self.normal_data.shape}, Fault={self.fault_data.shape}, Attack={self.attack_data.shape}")
        
        except Exception as e:
            logger.error(f" Error loading data: {e}")
            raise

    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def replace_all_blank(self, value):
        result = re.sub(r'\W+', ' ', value).replace("_", ' ')
        result = re.sub(r'\d', ' ', result)
        return result

    def remove_stop_words(self, text):
        stop_words = ['about', 'here', 'at', 'is', 'from', 'or', 'on', 'for', 'of', 'by', 'the']
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def lemmatize_stop(self, text):
        document = nlp(text)
        return " ".join([token.lemma_ for token in document])

    def clean_text(self, text):
        """Cleans and normalizes ICS log messages."""
        text = self.camel_to_snake(text)
        text = self.replace_all_blank(text)
        text = self.remove_stop_words(text)
        text = " ".join(self.lemmatize_stop(text))
        return text

        
    def event_to_number(self, text):
        
        text = self.clean_text(text) 
        if hasattr(self, 'event2template'):
            for event_id, event_template in self.event2template.items():
                if text == event_template:
                    return event_id
        return -1  




    def split_data(self):
        self.normal_data["label"] = 0
        self.fault_data["label"] = 1
        self.attack_data["label"] = 2

        normal_train, normal_test = train_test_split(self.normal_data, train_size=0.50, random_state=42)
        normal_test = normal_test.sample(frac=0.20, random_state=42)
        normal_val, normal_test = train_test_split(normal_test, test_size=0.80, random_state=42)

        fault_train, fault_test = train_test_split(self.fault_data, train_size=0.70, random_state=42)
        fault_val, fault_test = train_test_split(fault_test, test_size=0.80, random_state=42)

        attack_val, attack_test = train_test_split(self.attack_data, test_size=0.80, random_state=42)

        combined_val = pd.concat([normal_val, fault_val, attack_val], ignore_index=True)
        combined_test = pd.concat([normal_test, fault_test, attack_test], ignore_index=True)

        logger.info(f" Final Combined Splits: Validation={combined_val.shape}, Test={combined_test.shape}")

        return normal_train.drop(columns=["label"]), fault_train.drop(columns=["label"]), combined_val, combined_test
    
    def train_word2vec(self, corpus):
        processed_corpus = [gensim.utils.simple_preprocess(text) for text in corpus]

        if os.path.exists(self.w2v_model_path):
            logger.info(f" Loading existing Word2Vec model from {self.w2v_model_path}")
            try:
                self.w2v_model = Word2Vec.load(self.w2v_model_path)

                if self.w2v_model.vector_size != 100:
                    logger.warning(f"Model vector size mismatch! Expected 100, found {self.w2v_model.vector_size}. Retraining.")
                    raise ValueError("Vector size mismatch, retraining...")

                vocab_before = set(self.w2v_model.wv.key_to_index.keys())
                vocab_after = set(word for text in processed_corpus for word in text)
                new_words = vocab_after - vocab_before

                if new_words:
                    logger.info(f" Retraining Word2Vec with {len(new_words)} new words...")
                    self.w2v_model.build_vocab(processed_corpus, update=True)
                    self.w2v_model.train(processed_corpus, total_examples=len(processed_corpus), epochs=100)
                    self.w2v_model.save(self.w2v_model_path)
                    logger.info(" Retrained existing Word2Vec model.")
                else:
                    logger.info(" No new words found. Using existing Word2Vec model.")
                return
            except Exception as e:
                logger.error(f" Failed to load Word2Vec: {e}, retraining from scratch.")

        logger.info(" Training new full Word2Vec model...")
        self.w2v_model = Word2Vec(
            sentences=processed_corpus,
            vector_size=100, 
            window=7,  
            min_count=5,  
            workers=8,
            sg=1,  
            epochs=500
        )
        self.w2v_model.save(self.w2v_model_path)
        logger.info(f" Fully trained Word2Vec model saved at {self.w2v_model_path}")


    
    def compute_tfidf_weights(self, corpus):
        word_counts = Counter()
        total_documents = len(corpus)

        for text in corpus:
            unique_words = set(gensim.utils.simple_preprocess(text))
            for word in unique_words:
                word_counts[word] += 1

        self.word2idf = {word: np.log((total_documents + 1) / max(1, count)) + 1 for word, count in word_counts.items()}
        self.word2idf['oov'] = np.log((total_documents + 1) / 1) + 1  

        logger.info(f" Computed TF-IDF weights for {len(self.word2idf)} words.")



    def text_to_sequence(self, texts):
        sequences = []
        embedding_dim = self.w2v_model.vector_size  

        for text in texts:
            tokens = gensim.utils.simple_preprocess(text)
            sequence = []
            total_weight = 0

            for word in tokens:
                if word in self.w2v_model.wv:
                    idf = self.word2idf.get(word, self.word2idf['oov'])  
                    weighted_vector = self.w2v_model.wv[word] * idf  
                    sequence.append(weighted_vector)
                    total_weight += idf
                else:
                    sequence.append(np.random.uniform(-0.25, 0.25, embedding_dim))  

            if len(sequence) == 0:
                sequence = [np.zeros(embedding_dim)]  

            sequences.append(sequence)

        return sequences




    
    def compute_optimal_padding(self, sequences, label):
        sequence_lengths = [len(seq) for seq in sequences if len(seq) > 0]

        if len(sequence_lengths) == 0:
            logger.warning(f"No valid sequences found for {label}. Using minimum padding.")
            return 30  

        max_seq_length = int(np.percentile(sequence_lengths, 95))

        max_seq_length = min(30, max_seq_length)  

        logger.info(f" {label} - Optimal Padding Length: {max_seq_length}")
        return max_seq_length

    


    
    def preprocess_text(self, train_texts, val_texts, test_texts):
        logger.info(" Applying TF-IDF transformation...")
        train_cleaned = train_texts.apply(self.clean_text)
        val_cleaned = val_texts.apply(self.clean_text)
        test_cleaned = test_texts.apply(self.clean_text)

        self.vectorizer.fit(train_cleaned)
        X_train = self.vectorizer.transform(train_cleaned).toarray()
        X_val = self.vectorizer.transform(val_cleaned).toarray()
        X_test = self.vectorizer.transform(test_cleaned).toarray()

        return X_train, X_val, X_test


    def prepare_data(self):
        self.load_data()
        
        normal_train, fault_train, combined_val, combined_test = self.split_data()

        val_labels = combined_val["label"].values
        test_labels = combined_test["label"].values

        val_timestamps = combined_val["Timestamp"] if "Timestamp" in combined_val else None
        test_timestamps = combined_test["Timestamp"] if "Timestamp" in combined_test else None

        if val_timestamps is not None and val_timestamps.dropna().shape[0] > 0:
            val_timestamps.to_csv(os.path.join(self.output_dir, "val_timestamps.csv"), index=False)
            logger.info(" Validation timestamps saved.")
        else:
            logger.warning("No valid timestamps found for validation set.")

        if test_timestamps is not None and test_timestamps.dropna().shape[0] > 0:
            test_timestamps.to_csv(os.path.join(self.output_dir, "test_timestamps.csv"), index=False)
            logger.info(" Test timestamps saved.")
        else:
            logger.warning("No valid timestamps found for test set.")


        corpus = (
            normal_train["Message"].tolist() + 
            fault_train["Message"].tolist() + 
            combined_val["Message"].tolist() + 
            combined_test["Message"].tolist()
        )

        self.compute_tfidf_weights(corpus)
        self.train_word2vec(corpus)

        X_normal_train = self.text_to_sequence(normal_train["Message"])
        X_fault_train = self.text_to_sequence(fault_train["Message"])
        X_val = self.text_to_sequence(combined_val["Message"])
        X_test = self.text_to_sequence(combined_test["Message"])

       
        max_seq_length_normal = self.compute_optimal_padding(X_normal_train, "Normal Train")
        max_seq_length_fault = self.compute_optimal_padding(X_fault_train, "Fault Train")

        max_seq_length = max(max_seq_length_normal, max_seq_length_fault)
        embedding_dim = self.w2v_model.vector_size  
        X_normal_train_padded = pad_sequences(X_normal_train, maxlen=max_seq_length, padding="post", dtype="float32")
        X_fault_train_padded = pad_sequences(X_fault_train, maxlen=max_seq_length, padding="post", dtype="float32")
        X_val_padded = pad_sequences(X_val, maxlen=max_seq_length, padding="post", dtype="float32")
        X_test_padded = pad_sequences(X_test, maxlen=max_seq_length, padding="post", dtype="float32")



        X_normal_train_lstm = X_normal_train_padded.reshape((X_normal_train_padded.shape[0], max_seq_length, embedding_dim))
        X_fault_train_lstm = X_fault_train_padded.reshape((X_fault_train_padded.shape[0], max_seq_length, embedding_dim))
        X_val_lstm = X_val_padded.reshape((X_val_padded.shape[0], max_seq_length, embedding_dim))
        X_test_lstm = X_test_padded.reshape((X_test_padded.shape[0], max_seq_length, embedding_dim))

        
        X_normal_train_ocsvm = np.mean(X_normal_train_padded, axis=1)
        X_fault_train_ocsvm = np.mean(X_fault_train_padded, axis=1)
        X_val_ocsvm = np.mean(X_val_padded, axis=1)
        X_test_ocsvm = np.mean(X_test_padded, axis=1)
        
        
        logger.info(" Saved 2D feature vectors.")
        np.save(os.path.join(self.output_dir, "normal_train.npy"), X_normal_train_ocsvm)
        np.save(os.path.join(self.output_dir, "fault_train.npy"), X_fault_train_ocsvm)
        np.save(os.path.join(self.output_dir, "val.npy"), X_val_ocsvm)
        np.save(os.path.join(self.output_dir, "test.npy"), X_test_ocsvm)
        np.save(os.path.join(self.output_dir, "val_labels.npy"), val_labels)
        np.save(os.path.join(self.output_dir, "test_labels.npy"), test_labels)
        

        np.save(os.path.join(self.output_dir, "lstm_autoencoder_normal_train.npy"), X_normal_train_lstm)
        np.save(os.path.join(self.output_dir, "lstm_autoencoder_fault_train.npy"), X_fault_train_lstm)
        np.save(os.path.join(self.output_dir, "lstm_autoencoder_val.npy"), X_val_lstm)
        np.save(os.path.join(self.output_dir, "lstm_autoencoder_test.npy"), X_test_lstm)

        # Logging dataset details
        logger.info(f" Saved 2D processed datasets in {self.output_dir}:")
        logger.info(f"    Normal Train: { X_normal_train_ocsvm.shape} (Labels Dropped)")
        logger.info(f"    Fault Train: {X_fault_train_ocsvm.shape} (Labels Dropped)")
        logger.info(f"    Validation: { X_val_ocsvm.shape} (Labels: {val_labels.shape})")
        logger.info(f"    Test: {X_test_ocsvm.shape} (Labels: {test_labels.shape})")
        logger.info(f" Saved LSTM-Autoencoder 3D processed datasets in {self.output_dir}:")
        logger.info(f"    Normal Train: {X_normal_train_lstm .shape} (Labels Dropped)")
        logger.info(f"    Fault Train: {X_fault_train_lstm.shape} (Labels Dropped)")
        logger.info(f"    Validation: {X_val_lstm.shape} (Labels: {val_labels.shape})")
        logger.info(f"    Test: { X_test_lstm.shape} (Labels: {test_labels.shape})")
        logger.info(f"    Validation Timestamps: {'Saved' if val_timestamps is not None and not val_timestamps.isnull().all() else 'N/A'}")
        logger.info(f"    Test Timestamps: {'Saved' if test_timestamps is not None and not test_timestamps.isnull().all() else 'N/A'}")

        logger.info(f" LSTM Autoencoder datasets successfully saved in {self.output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICS Data Preprocessing Pipeline")
    parser.add_argument("--normal_file", type=str, required=True)
    parser.add_argument("--fault_file", type=str, required=True)
    parser.add_argument("--attack_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--w2v_model_path", type=str, default="word2vec.model")

    args = parser.parse_args()

    pipeline = ICSDataPipeline(
        normal_file=args.normal_file,
        fault_file=args.fault_file,
        attack_file=args.attack_file,
        output_dir=args.output_dir,
        w2v_model_path=args.w2v_model_path
    )
    pipeline.prepare_data()