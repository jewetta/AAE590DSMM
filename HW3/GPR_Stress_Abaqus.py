import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats

#load data
file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 3\DataSet_Stress_Import_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :3].values.astype(np.float32)
y = df.iloc[:, 3].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lower = 1e-5
upper = 1e5

kernel = C(1.0, (lower, upper)) * RBF([1.0, 1.0, 1.0], (lower, upper))

#train model
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=200,normalize_y=True,random_state=42)
gp.fit(X_train_scaled, y_train)

#nre prediction
X_new = np.array([[0.4, 1.0, 2.0]], dtype=np.float32)
X_new_scaled = scaler.transform(X_new)
y_pred, sigma = gp.predict(X_new_scaled, return_std=True)

print(f"Predicted yield strength: {y_pred[0]:.2f} MPa")
print(f"Uncertainty: {sigma[0]:.2f} MPa")
print("Learned kernel:", gp.kernel_)

#predict on test data set
y_test_pred = gp.predict(X_test_scaled)
#model metrics
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"RMSE: {rmse:.2f} MPa")
print(f"R² Score: {r2:.3f}")
print(f"Test MAE: {test_mae:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Stress (MPa)")
plt.ylabel("Predicted Stress (MPa)")
plt.title("Actual vs. Predicted Stress Values")
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - y_test_pred
std_residuals = residuals / np.std(residuals)
print(f"Standardized Residuals Mean: {np.mean(std_residuals):.3f}")
print(f"Standardized Residuals Std Dev: {np.std(std_residuals):.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, color="blue", alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Stress (MPa)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.show()

#5 k fold cv
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_list = []
cv_r2_list = []
cv_mae_list = []

for train_index, test_index in kf.split(X):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    gp_cv = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=200,normalize_y=True,random_state=42)
    gp_cv.fit(X_train_cv_scaled, y_train_cv)
    
    y_pred_cv = gp_cv.predict(X_test_cv_scaled)
    
    fold_rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
    fold_r2 = r2_score(y_test_cv, y_pred_cv)
    fold_mae = mean_absolute_error(y_test_cv, y_pred_cv)
    
    cv_rmse_list.append(fold_rmse)
    cv_r2_list.append(fold_r2)
    cv_mae_list.append(fold_mae)

avg_rmse = np.mean(cv_rmse_list)
avg_r2 = np.mean(cv_r2_list)
avg_mae = np.mean(cv_mae_list)

print("\n5-Fold Cross-Validation Results:")
print(f"Average RMSE: {avg_rmse:.2f} MPa")
print(f"Average R²: {avg_r2:.3f}")
print(f"Average MAE: {avg_mae:.4f}")
