import os
os.environ.setdefault("XGBOOST_BUILD_WITH_CUDA", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

from xgboost import XGBRegressor
X = pd.read_csv("cdf_values.csv", dtype=np.float32)
y_df = pd.read_parquet("mixed_distributions_with_cdf_150k_20f.parquet")
y = y_df["multivariate_empirical_cdf"].astype(np.float32)

n = min(len(X), len(y))
X = X.iloc[:n].reset_index(drop=True)
y = y.iloc[:n].reset_index(drop=True)
mask = y != 0.0
X = X[mask].reset_index(drop=True)
y = (y[mask].reset_index(drop=True) * 1000.0).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
base_est = XGBRegressor(
    device="cuda",            
    tree_method="hist", 
    objective="reg:squarederror",
    eval_metric="rmse",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1,               
    verbosity=1,
    early_stopping_rounds=50
)
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05],
}
grid = GridSearchCV(
    estimator=base_est,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=1, 
    verbose=1,
    refit=False
)
grid.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

best_params = grid.best_params_
print("Best parameters and best score:", best_params, grid.best_score_)

best_params = grid.best_params_
best_model = clone(base_est)           
best_model.set_params(**best_params)

best_model.fit(
    X_train,y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)
print("Best iteration and best score:", getattr(best_model, "best_iteration", None), getattr(best_model, "best_score", None))

y_pred =best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("mse:", mse)