import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_features(full_features, selected_features):
    # Encode categorical variables
    le_gender = LabelEncoder()
    full_features['gender'] = le_gender.fit_transform(full_features['gender'])

    le_label = LabelEncoder()
    full_features['label'] = le_label.fit_transform(full_features['label'])

    # Select features and labels
    X = full_features[selected_features]
    y = full_features['label']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
