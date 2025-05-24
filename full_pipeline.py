import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Data verification function
def check_data(file_path, name):
    print(f'\n===== {name} =====')
    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
        print(df.describe(include='all'))
        print(df.isnull().sum())
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# 2. Feature engineering function example
def feature_engineering(df):
    # Example: Adding derived variables for ACC, GYRO, EMG
    for axis in ['X', 'Y', 'Z']:
        df[f'acc_total_{axis}'] = df[[f'acc_{axis}_mean', f'acc_{axis}_std', f'acc_{axis}_max', f'acc_{axis}_min', f'acc_{axis}_rms']].sum(axis=1)
        df[f'gyro_total_{axis}'] = df[[f'gyro_{axis}_mean', f'gyro_{axis}_std', f'gyro_{axis}_max', f'gyro_{axis}_min', f'gyro_{axis}_rms']].sum(axis=1)
    # Example: emg_ratio = rms / mav
    if 'rms' in df.columns and 'mav' in df.columns:
        df['emg_ratio'] = df['rms'] / (df['mav'] + 1e-8)
    return df

# 3. Modeling and validation function
def modeling_and_validation(df, label_col='exo'):
    # Remove missing values
    df = df.dropna()
    # Label encoding
    if df[label_col].dtype == 'O':
        df[label_col] = df[label_col].astype(str).map({'True':1, 'False':0, '1':1, '0':0})
    # Split features/labels
    X = df.drop([label_col, 'Muscle', 'gender'], axis=1, errors='ignore')
    y = df[label_col].astype(int)
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Model and grid search
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    print(f"\nBest Params: {grid.best_params_}")
    y_pred = grid.predict(X_test_scaled)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    # Feature Importance visualization
    importances = grid.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    plt.title('Feature Importances')
    plt.bar(range(10), importances[indices[:10]], align='center')
    plt.xticks(range(10), [X.columns[i] for i in indices[:10]], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance_pipeline.png')
    plt.show()
    return grid, X_test, y_test, y_pred

def pattern_analysis(df):
    # 1. Class-wise distribution of important features
    important_cols = ['rms', 'mav', 'mean_freq', 'acc_X_mean', 'gyro_Y_mean', 'emg_ratio']
    for col in important_cols:
        if col in df.columns:
            plt.figure(figsize=(7,4))
            sns.boxplot(x='exo', y=col, data=df)
            plt.title(f'Class-wise distribution: {col}')
            plt.savefig(f'pattern_{col}.png')
            plt.close()
    # 2. Feature means by muscle
    if 'Muscle' in df.columns:
        muscle_means = df.groupby('Muscle')[[c for c in important_cols if c in df.columns]].mean()
        print('\n[Feature means by muscle]')
        print(muscle_means)
    # 3. Correlation heatmap
    corr_cols = [c for c in important_cols if c in df.columns]
    if len(corr_cols) > 1:
        plt.figure(figsize=(10,8))
        corr = df[corr_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation')
        plt.savefig('pattern_corr_heatmap.png')
        plt.close()

def check_sampling_sync(df):
    """
    Check if EMG data is properly downsampled to match IMU timestamps
    """
    print("\n===== Sampling Rate & Synchronization Check =====")
    
    # 1. Check time series consistency
    if 'IMU_TimeSeries' in df.columns:
        time_diff = df['IMU_TimeSeries'].diff()
        print("\n1. Time Series Analysis:")
        print(f"Average time step: {time_diff.mean():.6f} seconds")
        print(f"Time step std: {time_diff.std():.6f} seconds")
        print(f"Min time step: {time_diff.min():.6f} seconds")
        print(f"Max time step: {time_diff.max():.6f} seconds")
        
        # Plot time steps
        plt.figure(figsize=(10,4))
        plt.plot(time_diff.values)
        plt.title('Time Steps Between Samples')
        plt.xlabel('Sample Index')
        plt.ylabel('Time Step (seconds)')
        plt.savefig('time_steps_analysis.png')
        plt.close()
    
    # 2. Check EMG-IMU data points
    emg_cols = [col for col in df.columns if 'EMG' in col]
    imu_cols = [col for col in df.columns if 'ACC' in col or 'GYRO' in col]
    
    print("\n2. Data Points Analysis:")
    print(f"Total samples: {len(df)}")
    print(f"EMG columns: {len(emg_cols)}")
    print(f"IMU columns: {len(imu_cols)}")
    
    # 3. Check for missing values in EMG and IMU
    print("\n3. Missing Values Check:")
    emg_missing = df[emg_cols].isnull().sum().sum()
    imu_missing = df[imu_cols].isnull().sum().sum()
    print(f"EMG missing values: {emg_missing}")
    print(f"IMU missing values: {imu_missing}")
    
    # 4. Plot EMG and IMU signals for first 100 samples
    if len(df) > 100:
        plt.figure(figsize=(15,8))
        plt.subplot(2,1,1)
        plt.plot(df['IMU_TimeSeries'].iloc[:100], df[emg_cols[0]].iloc[:100], label='EMG')
        plt.title('EMG Signal (First 100 samples)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(df['IMU_TimeSeries'].iloc[:100], df[imu_cols[0]].iloc[:100], label='IMU')
        plt.title('IMU Signal (First 100 samples)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('signal_sync_check.png')
        plt.close()

def create_feature_heatmap(df):
    """
    Create a comprehensive feature correlation heatmap
    """
    print("\n===== Feature Correlation Analysis =====")
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find highly correlated features (absolute correlation > 0.5)
    high_corr_features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.5:
                high_corr_features.append(corr_matrix.columns[i])
                high_corr_features.append(corr_matrix.columns[j])
    
    high_corr_features = list(set(high_corr_features))
    
    # Create heatmap for highly correlated features
    plt.figure(figsize=(15,12))
    high_corr_matrix = corr_matrix.loc[high_corr_features, high_corr_features]
    sns.heatmap(high_corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Feature Correlation Heatmap (|correlation| > 0.5)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png')
    plt.close()
    
    # Print top correlations
    print("\nTop 10 Feature Correlations:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((corr_matrix.columns[i], 
                             corr_matrix.columns[j], 
                             corr_matrix.iloc[i,j]))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for pair in corr_pairs[:10]:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

if __name__ == '__main__':
    # 1. Data verification
    processed_df = check_data('processed_data.csv', 'Processed Data')
    features_df = check_data('extracted_features.csv', 'Extracted Features')
    
    # Check sampling synchronization
    if processed_df is not None:
        check_sampling_sync(processed_df)
    
    # 2. Feature engineering
    if features_df is not None:
        features_df = feature_engineering(features_df)
        pattern_analysis(features_df)
        create_feature_heatmap(features_df)
    # 3. Modeling and validation
    if features_df is not None:
        modeling_and_validation(features_df, label_col='exo') 