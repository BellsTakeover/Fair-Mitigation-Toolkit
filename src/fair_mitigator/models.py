# models.py
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def make_model(cfg: dict):
    """
    Builds a model from config.

    RF-only toolkit:
      model:
        kind: random_forest
        params:
          n_estimators: 300
          random_state: 42
          n_jobs: 1
          # optional "regularization" knobs:
          max_depth: null
          min_samples_leaf: 1
          min_samples_split: 2
          max_features: sqrt
          ccp_alpha: 0.0
    """
    kind = cfg.get("kind", "random_forest")
    if kind != "random_forest":
        raise ValueError("This toolkit is RF-only. Set model.kind to 'random_forest'.")

    p = cfg.get("params", {})

    return RandomForestClassifier(
        n_estimators=int(p.get("n_estimators", 300)),
        random_state=int(p.get("random_state", 42)),
        n_jobs=int(p.get("n_jobs", 1)),

        # "regularization" knobs for RF
        max_depth=p.get("max_depth", None),
        min_samples_leaf=int(p.get("min_samples_leaf", 1)),
        min_samples_split=int(p.get("min_samples_split", 2)),
        max_features=p.get("max_features", "sqrt"),
        ccp_alpha=float(p.get("ccp_alpha", 0.0)),

        # helps with imbalance without forcing SMOTE
        class_weight=p.get("class_weight", None),
    )


def run_grid_search(model, grid_cfg: dict, X_train, y_train):
    """
    grid_cfg schema:
      grid_search:
        enabled: true
        param_grid: {...}
        cv: 3
        scoring: accuracy
    """
    grid = GridSearchCV(
        model,
        grid_cfg["param_grid"],
        cv=int(grid_cfg.get("cv", 3)),
        scoring=str(grid_cfg.get("scoring", "accuracy")),
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

