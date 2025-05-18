# Exoskeleton Impact Analysis (Team Midterm Project)

## Overview
This project analyzes wearable sensor data from repetitive lifting tasks to assess the impact of exoskeleton use.
We extract EMG and IMU features and train classification models to distinguish between Exo and NoExo conditions.

## Data
- 4 CSV files:
  - P3_Exo, P3_NoExo, P4_Exo, P4_NoExo
- Sensor 1 (Right Bicep) used in the current analysis
- EMG, ACC, GYRO data only

## Methods
1. Preprocessing: cleaned data and removed time series columns
2. Feature Extraction: statistical features via rolling window (250 size / 125 step)
3. Modeling: Trained Logistic Regression, KNN, Decision Tree, and Random Forest
4. Evaluation: Confusion matrix, ROC curve, feature importance, and boxplot

## Feature Matrix

- After rolling window feature extraction, we construct(current):

Feature Matrix (X): shape = (n_samples, n_features)  
  - Each row: a 0.2second window of Right Bicep sensor data  
  - Each column: a computed feature (e.g., emg_mean, gyro_mean, accel_total)

Label Vector (y): shape = (n_samples,)  
  - Each element: either Exo or NoExo

- Example:  
  - X: (500, 13)  
  - y: (500,)  


## Key Findings
- Random Forest showed the highest performance
- Features like gyro_mean and accel_mean are most important

## Next?
- Analyze additional sensors: Right Delt, Left Bicep, Left Delt
- Explore multi-sensor feature fusion

## Reference
- Use of a Wearable Electromyography Armband to Detect Lift-Lower Tasks and Classify Hand Loads
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4743121
