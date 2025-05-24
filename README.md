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

## Model Evaluation and Hyperparameter Selection

### Model Selection
- **Algorithm**: Random Forest Classifier
  - Rationale: 
    - Handles non-linear relationships well
    - Robust to outliers and noise
    - Provides feature importance insights
    - Good performance with mixed feature types
    - Less prone to overfitting compared to single decision trees

### Hyperparameter Selection
- **Grid Search Parameters**:
  ```python
  param_grid = {
      'n_estimators': [50, 100],      # Number of trees
      'max_depth': [None, 10, 20],    # Tree depth
      'min_samples_leaf': [1, 2],     # Minimum samples per leaf
      'max_features': ['sqrt', 'log2'] # Features to consider for splits
  }
  ```
- **Selection Rationale**:
  1. `n_estimators`: 
     - Limited to 50-100 to balance performance and computation time
     - Higher values showed diminishing returns
  2. `max_depth`: 
     - None: Allow full tree growth
     - 10, 20: Prevent overfitting
  3. `min_samples_leaf`: 
     - 1: Allow fine-grained splits
     - 2: Prevent overfitting
  4. `max_features`: 
     - 'sqrt': Standard choice for classification
     - 'log2': Alternative for high-dimensional data

### Evaluation Metrics
1. **Primary Metrics**:
   - Accuracy: 94.15%
   - F1 Score: 94.18%
   - Rationale: 
     - F1 score chosen as primary metric due to class balance importance
     - Combines precision and recall effectively

2. **Confusion Matrix Analysis**:
   ```
   [[88  4]
    [ 7 89]]
   ```
   - True Negatives: 88
   - False Positives: 4
   - False Negatives: 7
   - True Positives: 89
   - Shows balanced performance across classes

3. **Cross-Validation**:
   - 3-fold cross-validation used
   - Ensures robust performance estimation
   - Prevents overfitting

### Feature Importance
- Top 5 important features:
  1. Y-axis gyroscope mean
  2. X-axis accelerometer mean
  3. EMG RMS
  4. EMG MAV
  5. EMG WL

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
