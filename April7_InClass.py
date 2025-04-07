import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (GridSearchCV,
                                     KFold,
                                     train_test_split)

# creating a dummy data set
X, y = make_regression(n_samples=100)

# train test split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, shuffle=False, test_size=0.2)

# cv procedure
cv = KFold(n_splits=5)

# defining the search space
model = RandomForestRegressor()
param_search = {'n_estimators': [10, 50, 100]}

# applying CV with a gridsearch on the training data
gs = GridSearchCV(estimator=model,
                  cv=cv,
                  param_grid=param_search)

# fit the model
gs.fit(X_train, y_train)

# re-training the best approach for testing
chosen_model = RandomForestRegressor(**gs.best_params_)
chosen_model.fit(X_train, y_train)

# inference on test set and evaluation
preds = chosen_model.predict(X_test)

# compute metrics
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds)

# print metrics
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# final model for deployment
final_model = RandomForestRegressor(**gs.best_params_)
final_model.fit(X, y)
