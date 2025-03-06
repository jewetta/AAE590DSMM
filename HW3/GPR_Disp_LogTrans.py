import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats
from sklearn.model_selection import KFold

# Load data
file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 3\DataSet_Disp_Import_csv.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.iloc[:, :3].values.astype(np.float32)
y = df.iloc[:, 3].values.astype(np.float32)

# Apply log transform to the dependent variable (stress)
y_log = np.log(y)

# Split data into train and test sets (using log-transformed y)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lower = 1e-5
upper = 1e5
# Define kernel (optimized with your kernel optimizer)
kernel = C(1.0, (lower, upper)) * Matern(
    length_scale=[1.0, 1.0, 1.0],
    length_scale_bounds=(lower, upper),
    nu=2.5
)
#kernel = C(1.0, (lower, upper)) * RBF([1.0, 1.0, 1.0], (lower, upper))
# Create and fit Gaussian Process model on log(stress)
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=200,
    normalize_y=True,
    random_state=42
)
gp.fit(X_train_scaled, y_train_log)

# Predict for a new composition
X_new = np.array([[0.55, 5, 2.0]], dtype=np.float32)
X_new_scaled = scaler.transform(X_new)
y_pred_log, sigma_log = gp.predict(X_new_scaled, return_std=True)
# Exponentiate to get predictions on the original scale.
y_pred = np.exp(y_pred_log)
# Approximate uncertainty propagation using the delta method:
sigma = y_pred * sigma_log

print(f"Predicted yield strength: {y_pred[0]:.2f} mm")
print(f"Uncertainty: {sigma[0]:.2f} mm")
print("Learned kernel:", gp.kernel_)

# Predict on the test set (in log space) and then transform back
y_test_pred_log = gp.predict(X_test_scaled)
y_test_pred = np.exp(y_test_pred_log)
y_test_orig = np.exp(y_test_log)

# Compute metrics on original scale
rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
r2 = r2_score(y_test_orig, y_test_pred)
test_mae = mean_absolute_error(y_test_orig, y_test_pred)

print(f"RMSE: {rmse:.2f} mm")
print(f"R² Score: {r2:.3f}")
print(f"Test MAE: {test_mae:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_test_pred, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)],
         color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Disp (mm)")
plt.ylabel("Predicted Stress (mm)")
plt.title("Actual vs. Predicted Disp Values")
plt.legend()
plt.grid(True)
plt.show()

# Residual analysis
residuals = y_test_orig - y_test_pred
std_residuals = residuals / np.std(residuals)
print(f"Standardized Residuals Mean: {np.mean(std_residuals):.3f}")
print(f"Standardized Residuals Std Dev: {np.std(std_residuals):.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, color="blue", alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Disp (mm)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.show()

#5 k fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_list = []
cv_r2_list = []
cv_mae_list = []

for train_index, test_index in kf.split(X):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index] 
    
    y_train_cv_log = np.log(y_train_cv)
    
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    gp_cv = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=200,normalize_y=True,random_state=42)
    gp_cv.fit(X_train_cv_scaled, y_train_cv_log)
    
    y_pred_cv_log = gp_cv.predict(X_test_cv_scaled)
    y_pred_cv = np.exp(y_pred_cv_log)  
    
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
print(f"Average RMSE: {avg_rmse:.2f} mm")
print(f"Average R²: {avg_r2:.3f}")
print(f"Average MAE: {avg_mae:.4f}")
