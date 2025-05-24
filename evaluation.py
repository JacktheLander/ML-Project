import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd

# Print classification report
def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"=== {model_name} ===")
    print(classification_report(y_true, y_pred))

# Plot feature importances for tree-based models
def plot_feature_importance(model, feature_names, title="Feature Importance"):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feat_imp, y=feat_imp.index)
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
