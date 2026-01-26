#makes the models, i.e random forest
#, i.e.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def make_model(cfg):
    if cfg["kind"] == "random_forest":
        return RandomForestClassifier(**cfg["params"])
    raise ValueError("Unknown model type")

def run_grid_search(model, grid_cfg, X_train, y_train):
    grid = GridSearchCV(
        model,
        grid_cfg["param_grid"],
        cv=grid_cfg["cv"],
        scoring=grid_cfg["scoring"]
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
