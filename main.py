from filters import bandpass_filter_emg, lowpass_filter_imu
from feature_extraction import extract_features_sliding_window
from feature_engineering import add_derived_features
from preprocessing import preprocess_features
from modeling import define_models
from evaluation import evaluate_model, plot_feature_importance

import pandas as pd
import numpy as np
from data_loader import read_run, column_clean

# Fully working implementation of process_all_files with strict handling
def process_all_files():
    files = [
        ("/content/P3_Exo_1_0.csv", 2, 'male', 'Exo'),
        ("/content/P3_NoExo_1_0.csv", 1, 'male', 'NoExo'),
        ("/content/P4_Exo_1_0.csv", 2, 'female', 'Exo'),
        ("/content/P4_NoExo_1_0.csv", 1, 'female', 'NoExo')
    ]

    results = []
    for path, run, gender, label in files:
        df = read_run(path)
        df = column_clean(df, run, gender)
        df = df[[col for col in df.columns if col not in ['Time (s)', 'Timestamp']]]
        for col in ['EMG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        df['Filtered_EMG'] = bandpass_filter_emg(df['EMG'])
        for axis in ['X', 'Y', 'Z']:
            df[f'Filtered_ACC_{axis}'] = lowpass_filter_imu(df[f'ACC_{axis}'])
            df[f'Filtered_GYRO_{axis}'] = lowpass_filter_imu(df[f'GYRO_{axis}'])

        feats = extract_features_sliding_window(
            df,
            emg_col='Filtered_EMG',
            acc_cols=[f'Filtered_ACC_{a}' for a in ['X', 'Y', 'Z']],
            gyro_cols=[f'Filtered_GYRO_{a}' for a in ['X', 'Y', 'Z']],
            label=label,
            gender=gender
        )
        results.append(feats)

    return pd.concat(results, ignore_index=True)

def main():
    # Load and preprocess data
    full_features = process_all_files()
    full_features = add_derived_features(full_features)

    selected_features = [
        'emg_max', 'emg_min', 'emg_rms', 'emg_range', 'emg_variance',
        'acc_peak', 'acc_range', 'acc_std', 'acc_energy',
        'gyro_peak', 'gyro_range', 'gyro_std', 'gyro_energy',
        'emg_fft_mean_freq', 'emg_fft_median_freq', 'emg_fft_power',
        'gender'
    ]

    X_train, X_test, y_train, y_test = preprocess_features(full_features, selected_features)

    # Define and train models
    xgb, rf, lr, voting_clf, stacking_clf = define_models()
    models = {
        'XGBoost': xgb,
        'Random Forest': rf,
        'Voting': voting_clf,
        'Stacking': stacking_clf
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, model_name=name)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, selected_features, title=f"{name} Feature Importance")

if __name__ == "__main__":
    main()
