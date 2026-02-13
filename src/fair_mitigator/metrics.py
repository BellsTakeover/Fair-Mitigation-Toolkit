import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

def evaluate(y_true, y_pred):
    """
    Standard evaluation metrics returned as a dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
    }

def cross_validate(estimator, X, y, cv_cfg, random_state):
    """
    K-fold cross-validation for reporting robustness.
    """
    kf = KFold(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=random_state,
    )

    scoring = cv_cfg.get("scoring", "accuracy")
    scores = cross_val_score(estimator, X, y, cv=kf, scoring=scoring)

    return {
        "cv_scores": scores.tolist(),
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_scoring": scoring,
    }
