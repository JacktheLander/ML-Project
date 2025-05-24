from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier

def define_models(best_xgb=None):
    if best_xgb is None:
        best_xgb = XGBClassifier(
            learning_rate=0.2, max_depth=5, n_estimators=200,
            subsample=0.8, eval_metric='logloss', use_label_encoder=False, random_state=42
        )
    rf = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('xgb', best_xgb), ('rf', rf), ('lr', lr)],
        voting='soft'
    )

    stacking_clf = StackingClassifier(
        estimators=[('xgb', best_xgb), ('rf', rf), ('lr', lr)],
        final_estimator=LogisticRegression(),
        passthrough=True,
        cv=5
    )

    return best_xgb, rf, lr, voting_clf, stacking_clf
