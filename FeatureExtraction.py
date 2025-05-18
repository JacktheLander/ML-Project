import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import seaborn as sns



# Define the CSV file paths and associated conditions (Exo/NoExo)
file_paths = [
    ("P3_Exo", "/content/P3_Exo_1_0.csv", "Exo"),
    ("P3_NoExo", "/content/P3_NoExo_1_0.csv", "NoExo"),
    ("P4_Exo", "/content/P4_Exo_1_0.csv", "Exo"),
    ("P4_NoExo", "/content/P4_NoExo_1_0.csv", "NoExo")
]

# Column names for Sensor 1 (Right Bicep)
SENSOR1_COLS = ['EMG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']

# Load CSV files, extract Sensor 1 columns, and clean missing values
def load_and_prepare():
    data = []
    for name, path, label in file_paths:
        try:
            df = pd.read_csv(path, skiprows=8, header=None, low_memory=False)
            df = df.iloc[:, [1, 3, 5, 7, 9, 11, 13]].copy()
            df.columns = SENSOR1_COLS
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            data.append((name, df, label))
        except Exception as e:
            print(f"Error loading {name}: {e}")
    return data


# Feature extraction functions

# EMG features: mean, max, min, std, RMS
def compute_emg_features(signal):
    return {
        'emg_mean': np.mean(signal),
        'emg_max': np.max(signal),
        'emg_min': np.min(signal),
        'emg_std': np.std(signal),
        'emg_rms': np.sqrt(np.mean(signal**2))
    }

# Accelerometer features: magnitude-based stats
def compute_accel_features(a_x, a_y, a_z):
    a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    return {
        'accel_peak': np.max(a_mag),
        'accel_mean': np.mean(a_mag),
        'accel_total': np.sqrt(np.mean(a_x**2) + np.mean(a_y**2) + np.mean(a_z**2)),
        'accel_range': np.max(a_mag) - np.min(a_mag)
    }

# Gyroscope features: magnitude-based stats
def compute_gyro_features(w_x, w_y, w_z):
    w_mag = np.sqrt(w_x**2 + w_y**2 + w_z**2)
    return {
        'gyro_peak': np.max(w_mag),
        'gyro_mean': np.mean(w_mag),
        'gyro_total': np.sqrt(np.mean(w_x**2) + np.mean(w_y**2) + np.mean(w_z**2)),
        'gyro_range': np.max(w_mag) - np.min(w_mag)
    }

# Extract rolling window features for EMG, ACC, and GYRO signals
def extract_features_with_rolling_window(df, window_size=250, step_size=125):
    features = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start+window_size]
        feats = {}
        feats.update(compute_emg_features(window['EMG']))
        feats.update(compute_accel_features(window['ACC_X'], window['ACC_Y'], window['ACC_Z']))
        feats.update(compute_gyro_features(window['GYRO_X'], window['GYRO_Y'], window['GYRO_Z']))
        features.append(feats)
    return pd.DataFrame(features)

# Run feature extraction pipeline for all subjects
data_entries = []
for name, df_clean, label in load_and_prepare():
    print(f"{name} cleaned length: {len(df_clean)}")
    feats = extract_features_with_rolling_window(df_clean)
    feats['subject'] = name
    feats['condition'] = label
    data_entries.append(feats)

# Combine extracted features into a single DataFrame
final_df = pd.concat(data_entries, ignore_index=True)
print(final_df.head())

# Split features and target
X = final_df.drop(columns=['subject', 'condition'])
y = final_df['condition']

# Train-test split (70% train, 30% test), stratified by label
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train label distribution:\n{y_train.value_counts()}\n")
print(f"Test label distribution:\n{y_test.value_counts()}\n")




# Define classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = (acc, cm)

    # Print evaluation results
    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print()

    # Plot confusion matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()




# Train Random Forest model to assess feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract feature importance from trained model
importances = rf.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Extract feature importance from trained model
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()




# Select top features for condition-wise comparison
top_features = [
    'gyro_mean', 'accel_mean', 'gyro_total', 'gyro_range', 'gyro_peak'
]

# Draw boxplots to compare Exo vs NoExo for top features
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=final_df, x='condition', y=feature)
    plt.title(f"{feature} by Condition (Exo vs NoExo)")
    plt.xlabel("Condition")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

# Initialize the Random Forest classifier with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data
rf_model.fit(X_train, y_train)

# Predict class probabilities on the test set
y_prob_rf = rf_model.predict_proba(X_test)

# Encode string labels into binary values (e.g., 'Exo' = 1, 'NoExo' = 0)
le = LabelEncoder()
y_test_bin = le.fit_transform(y_test)

# Identify the index of the positive class (usually 'Exo')
pos_class_index = list(rf_model.classes_).index(le.classes_[1])

# Extract the predicted probabilities for the positive class
y_scores = y_prob_rf[:, pos_class_index]

# Compute the False Positive Rate and True Positive Rate for ROC curve
fpr, tpr, _ = roc_curve(y_test_bin, y_scores)

# Calculate the Area Under the Curve (AUC) score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Exo vs NoExo)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
