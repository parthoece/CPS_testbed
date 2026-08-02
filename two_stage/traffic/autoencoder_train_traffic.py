import os
import logging
import numpy as np
import joblib
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

log_file = "autoencoder_train_traffic.log"

logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f" Starting autoencoder on traffic data")
"""
def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder
"""
def build_autoencoder(input_dim=100, encoding_dim=48):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

def train_autoencoder_model(train_data, encoding_dim):
    model = build_autoencoder(train_data.shape[1], encoding_dim)
    model.fit(train_data, train_data,
              epochs=100,
              batch_size=32,
              shuffle=True,
              validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=0)
    return model


def tune_threshold(model, val_data, val_labels, output_dir, stage, trial_number):
    recon_errors = np.mean(np.square(val_data - model.predict(val_data, verbose=0)), axis=1)
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(recon_errors.reshape(-1, 1)).flatten()

    thresholds = np.linspace(0, 1, 100)
    f1_scores, precision_scores, recall_scores, y_preds = [], [], [], []

    for t in thresholds:
        y_pred = np.where(scores >= t, 1, 0)
        f1_scores.append(f1_score(val_labels, y_pred))
        precision_scores.append(precision_score(val_labels, y_pred, zero_division=0))
        recall_scores.append(recall_score(val_labels, y_pred, zero_division=0))
        y_preds.append(y_pred)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    cm = confusion_matrix(val_labels, y_preds[best_idx])
    cm_pct = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100

    logging.info(f"Stage {stage} - Trial {trial_number} - Best Threshold: {best_thresh:.4f}, F1: {f1_scores[best_idx]:.4f}")
    logging.info(f"Confusion Matrix (Percentage):\n{np.round(cm_pct, 2)}")
    return best_thresh, f1_scores[best_idx], precision_scores[best_idx], recall_scores[best_idx], cm_pct


def evaluate_autoencoder(model, test_data, test_labels, threshold):
    recon_errors = np.mean(np.square(test_data - model.predict(test_data, verbose=0)), axis=1)
    y_pred = np.where(recon_errors >= threshold, 1, 0)
    f1 = f1_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall = recall_score(test_labels, y_pred, zero_division=0)
    cm = confusion_matrix(test_labels, y_pred)
    cm_pct = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0) * 100
    return f1, precision, recall, cm_pct, y_pred


def train_and_validate_stage(train_data, val_data, val_labels, test_data, test_labels, output_dir, stage, n_trials=10):
    os.makedirs(output_dir, exist_ok=True)
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial_output_dir = os.path.join(output_dir, f"stage{stage}_trial_{trial.number}")
        os.makedirs(trial_output_dir, exist_ok=True)

        encoding_dim = trial.suggest_int("encoding_dim", 8, 128)
        model = build_autoencoder(train_data.shape[1], encoding_dim)
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        logging.info(f"Stage {stage} - Trial {trial.number} - Params: encoding_dim={encoding_dim}" + " ".join(model_summary))
        encoding_dim = trial.suggest_int("encoding_dim", 8, 128)
        model = train_autoencoder_model(train_data, encoding_dim)
        joblib.dump(model, os.path.join(trial_output_dir, f"model_stage{stage}_trial{trial.number}.keras"))

        threshold, f1, prec, rec, _ = tune_threshold(model, val_data, val_labels, output_dir, stage, trial.number)
        return 0.5 * f1 + 0.25 * prec + 0.25 * rec

    study.optimize(objective, n_trials=n_trials)
    best_dim = study.best_params['encoding_dim']
    best_model = train_autoencoder_model(train_data, best_dim)
    threshold, f1, prec, rec, cm_pct = tune_threshold(best_model, val_data, val_labels, output_dir, stage, study.best_trial.number)
    f1, prec, rec, cm_pct, y_pred = evaluate_autoencoder(best_model, test_data, test_labels, threshold)

    best_model.save(os.path.join(output_dir, f"stage{stage}_autoencoder_model.keras"))
    return best_model, threshold, y_pred


def filter_stage2_data(model, threshold, val_data, test_data, val_labels, test_labels):
    val_scores = np.mean(np.square(val_data - model.predict(val_data, verbose=0)), axis=1)
    test_scores = np.mean(np.square(test_data - model.predict(test_data, verbose=0)), axis=1)

    val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min())
    test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min())

    val_idx = val_scores >= threshold
    test_idx = test_scores >= threshold

    val_data_filtered = val_data[val_idx]
    test_data_filtered = test_data[test_idx]
    val_labels_filtered = val_labels[val_idx]
    test_labels_filtered = test_labels[test_idx]

    val_labels_filtered[val_labels_filtered == 1] = 0
    val_labels_filtered[val_labels_filtered == 2] = 1
    test_labels_filtered[test_labels_filtered == 1] = 0
    test_labels_filtered[test_labels_filtered == 2] = 1

    return val_data_filtered, test_data_filtered, val_labels_filtered, test_labels_filtered


def main():
    base_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/ul/train/preprocess/traffic/split"
    normal_train_data = np.load(os.path.join(base_dir, "normal_train_minmax_scaled.npy"))
    fault_train_data = np.load(os.path.join(base_dir, "fault_train_minmax_scaled.npy"))
    val_data = np.load(os.path.join(base_dir, "final_validation_minmax_scaled.npy"))
    test_data = np.load(os.path.join(base_dir, "final_test_minmax_scaled.npy"))
    val_labels = np.load(os.path.join(base_dir, "final_validation_labels.npy")).ravel()
    test_labels = np.load(os.path.join(base_dir, "final_test_labels.npy")).ravel()

    val_labels_stage1 = np.where(val_labels > 0, 1, 0)
    test_labels_stage1 = np.where(test_labels > 0, 1, 0)

    output_dir_stage1 = os.path.join(base_dir, "stage1_output_autoencoder")
    output_dir_stage2 = os.path.join(base_dir, "stage2_output_autoencoder")

    model_stage1, threshold_stage1, pred_stage1 = train_and_validate_stage(
        normal_train_data, val_data, val_labels_stage1, test_data, test_labels_stage1, output_dir_stage1, stage=1
    )

    val_data_stage2, test_data_stage2, val_labels_stage2, test_labels_stage2 = filter_stage2_data(
        model_stage1, threshold_stage1, val_data, test_data, val_labels.copy(), test_labels.copy())

    model_stage2, threshold_stage2, pred_stage2 = train_and_validate_stage(
        fault_train_data, val_data_stage2, val_labels_stage2, test_data_stage2, test_labels_stage2, output_dir_stage2, stage=2
    )

    logging.info(" Autoencoder-based two-stage anomaly detection pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
