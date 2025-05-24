# EMG-IMU Data Analysis Pipeline

This project provides a comprehensive pipeline for analyzing and classifying EMG (Electromyography) and IMU (Inertial Measurement Unit) data.

## Project Structure

```
.
├── full_pipeline.py      # Main analysis pipeline
├── dataCleaning.py       # Data preprocessing module
├── feature_extraction.py # Feature extraction module
├── resampling.py         # Resampling module
├── processed_data.csv    # Preprocessed data
├── extracted_features.csv # Extracted features
└── Original Data Files
    ├── P3_Exo_1_0.csv
    ├── P3_NoExo_1_0.csv
    ├── P4_Exo_1_0.csv
    └── P4_NoExo_1_0.csv
```

## Key Features

### 1. Data Preprocessing (`dataCleaning.py`)
- EMG and IMU data cleaning
- Missing value handling
- Signal filtering (EMG: bandpass, IMU: lowpass)

### 2. Feature Extraction (`feature_extraction.py`)
- EMG features: RMS, MAV, WL, SSC, entropy, etc.
- IMU features: Accelerometer and gyroscope statistics
- Frequency domain features

### 3. Resampling (`resampling.py`)
- EMG and IMU data synchronization
- Rolling window-based downsampling

### 4. Analysis Pipeline (`full_pipeline.py`)
- Data loading and validation
- Feature engineering
- Model training and evaluation
- Result visualization

## Usage

1. Data Preparation
   - Place original data files in the project directory

2. Run Pipeline
   ```bash
   python full_pipeline.py
   ```

3. Check Results
   - `processed_data.csv`: Preprocessed data
   - `extracted_features.csv`: Extracted features
   - Generated image files:
     - `feature_correlation_heatmap.png`: Feature correlations
     - `pattern_*.png`: Class-wise feature distributions
     - `confusion_matrix.png`: Classification results

## Key Analysis Results

### Feature Importance
- Top 5 important features:
  1. Y-axis gyroscope mean
  2. X-axis accelerometer mean
  3. EMG RMS
  4. EMG MAV
  5. EMG WL

### Model Performance
- Accuracy: 94.15%
- F1 Score: 94.18%
- Balanced class performance

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

## Notes
- Be cautious of memory usage during data preprocessing
- Ensure sufficient disk space for large data processing 