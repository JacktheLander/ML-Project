# Function Descriptions

## 1. dataCleaning.py

  🔹 read_run(filename, skiprows=7)
Reads a raw CSV file containing EMG and IMU sensor data. It skips the first 7 rows, which include metadata and sensor frequency/cycle time information. The function extracts the first 56 columns and assigns meaningful column names based on body parts and sensor types. It returns a structured DataFrame containing the cleaned raw data.

🔹 column_clean(df)
Cleans the raw DataFrame by removing unnecessary time series columns, keeping only one representative time column for EMG and one for IMU. It also renames these columns for clarity and standardizes missing values by replacing empty strings, "NA", and None with np.nan.

🔹 create_sensor_col(df, run_num, gender, exo)
Reshapes the DataFrame using tidy_emg_imu_as_measured, transforming it from wide to long format with appropriate sensor labeling. It adds metadata columns: gender, run_num, and exo (the target variable). The reshaped DataFrame is saved to pivoted_df.csv and returned.

🔹 preprocessing_actions(full_df)
Prepares the data for machine learning by defining numerical and categorical features. Numerical data is imputed using the median and standardized, while categorical data is imputed using the most frequent value and one-hot encoded. The function combines these transformations using a ColumnTransformer, splits the dataset into training and test sets, applies the preprocessing, and returns the processed datasets and the fitted pipeline.

## 2. feature_extraction.py

🔹 extract_features(df)
This is the main feature extraction function that operates on a DataFrame containing filtered EMG, accelerometer, and gyroscope data. The data is grouped by the columns BodyPart, run_num, gender, and exo to compute features for each unique condition.

For accelerometer data (ACC X/Y/Z (G)_filtered), the function calculates the peak magnitude, mean magnitude, total acceleration (as the root of summed means of squares), and the range (max - min) of the acceleration magnitude vector.

For gyroscope data (GYRO X/Y/Z (deg/s)_filtered), it computes the peak angular velocity, mean angular velocity, total angular velocity (similar root sum of squares of means), and the range of the magnitude vector.

For EMG signals (EMG_MilliVolts_filtered), it extracts the mean, max, min, standard deviation, and RMS (root mean square). All features are stored in a list of dictionaries, which are then returned as a new DataFrame representing one row per group.

## 3. filtering.py

🔹 bandpass_filter_emg(series_signal, fs=1259, lowcut=20, highcut=450, order=4)
This function applies a band-pass Butterworth filter to EMG signals. It is designed to retain frequencies between 20 Hz and 450 Hz, which are relevant for muscle activity, while filtering out lower-frequency drift and higher-frequency noise. The function checks for edge cases such as empty or fully missing input data. It uses a zero-phase filter (filtfilt) to avoid phase distortion. The default sampling rate for EMG data is assumed to be 1259 Hz.

🔹 lowpass_filter_imu(series_signal, fs=148, cutoff=20, order=4)
This function applies a low-pass Butterworth filter to IMU sensor signals, typically accelerometer or gyroscope data. It retains frequencies below 20 Hz, which are most relevant for human motion, and removes high-frequency noise. Like the EMG filter, it includes edge case handling and uses zero-phase filtering. The default sampling rate for IMU data is set to 148 Hz.

## 4. main.py

🔹 overall_cleaning()
This function executes the complete sequence of steps required to load, clean, filter, reshape, and extract features from raw sensor data:

Data Loading: Four datasets are read using the read_run function, each representing different test conditions (with/without exoskeleton, male/female subjects).

Column Cleaning: The column_clean function is applied to remove redundant time columns and standardize missing values.

Downsampling: EMG data is downsampled using the downsample function to match the IMU sampling rate, allowing for synchronized processing.

Reshaping: Each cleaned DataFrame is transformed into a long format using create_sensor_col, which includes metadata such as run_num, gender, and exo.

Combining Data: All reshaped DataFrames are concatenated into a single combined dataset.

Signal Filtering:

IMU signals are low-pass filtered (<20 Hz) using lowpass_filter_imu for each axis (ACC and GYRO).

EMG signals are band-pass filtered (20–450 Hz) using bandpass_filter_emg.

Feature Extraction: The extract_features function is called to compute statistical features for EMG and IMU signals on a per-group basis.

Preprocessing for Machine Learning: The dataset is split into training and testing sets, and preprocessing pipelines (scaling, imputation, encoding) are applied using preprocessing_actions.

The function returns the fully processed and filtered combined DataFrame, ready for modeling or further analysis.

🔹 main()
A simple main function that runs the full pipeline when the script is executed directly.

## 5. modify.py

🔹 tidy_emg_imu_as_measured(df)
This function restructures a DataFrame that contains raw sensor readings from multiple body parts.

Identify Sensor Columns: It first identifies all columns associated with specific body parts (e.g., RDelt, LDelt, RBicep, LBicep) using string pattern matching.

Melt to Long Format: The identified measurement columns are melted into long format, where each row represents a single measurement from a specific body part and signal type.

Extract Body Part and Signal Type: New columns are created by splitting the original column names into BodyPart (e.g., RDelt) and Signal (e.g., EMG_MilliVolts, ACC X (G)).

Pivot Back to Wide Format: The data is then pivoted so that each signal type becomes its own column again, grouped by all original identifying variables plus the new BodyPart.

Final Formatting: Column names are flattened to ensure a clean, single-level header.

## 6. resample.py

🔹 downsample(df)
This function adjusts the sampling rate of high-frequency EMG signals to match the lower sampling rate of IMU signals (approximately 148 Hz). This ensures synchronization across sensor modalities for accurate multimodal analysis.

Key operations:

Numeric Conversion: All columns are converted to numeric types, with non-numeric values coerced to NaN, except time-related columns.

Time Indexing: EMG time values are converted to a timedelta format and used to set the DataFrame index for time-based resampling.

Resampling: EMG columns are resampled to 6.75ms intervals (corresponding to 148 Hz) using .resample(...).asfreq().

Interpolation and Filling: After resampling, missing EMG values are linearly interpolated. Any remaining NaNs are forward and backward filled.

Trimming: To ensure data alignment, the function locates the last valid index in the IMU columns and truncates the DataFrame to that point.

Output: Returns a cleaned, synchronized DataFrame with EMG signals downsampled and aligned with IMU data.
