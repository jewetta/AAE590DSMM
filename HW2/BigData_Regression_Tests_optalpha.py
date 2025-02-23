import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats

#to find optimal alpha value that min the MSE for both ridge and lasso models

file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 2\PropertySpace_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :4].values  
y = df.iloc[:, 4].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = np.logspace(-4, 1, 100)  

lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha_lasso = lasso_cv.alpha_

ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train)
best_alpha_ridge = ridge_cv.alpha_

print(f"Optimal alpha for Lasso: {best_alpha_lasso:.6f}")
print(f"Optimal alpha for Ridge: {best_alpha_ridge:.6f}")

