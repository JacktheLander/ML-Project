# Wearable Sensor Ergonomic Risk Assessment

This project analyzes EMG and IMU sensor data collected from participants performing lifting tasks with and without an exoskeleton. The goal is to extract biomechanical features and train machine learning models to assess the physical risk and impact of exoskeleton usage.


## Project Structure

### FeatureExtraction.py
Defines feature extraction functions for EMG, accelerometer, and gyroscope data.

#### Functions
- `compute_emg_features`: Calculates mean, max, min, std, and RMS of EMG signals.
- `compute_accel_features`: Computes peak, mean, and total acceleration, and range.
- `compute_gyro_features`: Computes peak, mean, and total angular velocity, and range.


### UpsamplingIMU.py
Handles resampling of IMU signals to match EMG sampling frequency (~1259 Hz).

#### Function
- `upsample`: Resamples IMU data using linear interpolation to synchronize with EMG for aligned windowing.


### dataCleaning.py
Handles loading and cleaning of raw CSV files for different participants and runs.

#### Functions
- `read_run`: Loads and renames columns for raw CSV data.
- `column_clean`: Cleans time series redundancy and adds labels for gender/run.
- `preprocessing`: Placeholder for additional data transformations.


### MainPipeline.py

Coordinates the full processing pipeline from loading to feature extraction.

#### Functions
- `overall_cleaning`: Loads and applies initial feature extraction on full signals.
- `extract_features_with_rolling_window`: Generates rolling window-based features for ML.
- `create_labeled_rolling_features`: Combines rolling features from all files and adds labels.

#### Output
- DataFrame with rolling window features and Exo/NoExo labels.
- Class distribution and sample preview printed on run.


## Usage
!python MainPipeline.py
ready for modeling.
