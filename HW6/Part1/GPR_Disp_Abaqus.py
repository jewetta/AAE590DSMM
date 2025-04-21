import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats
from sklearn.model_selection import KFold

#OLD FROM HW 3

#load data
file_path = r"/Users/austinjewett/Desktop/Purdue/AAE590/HW 3/DataSet_Disp_Import_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :3].values.astype(np.float32)
y = df.iloc[:, 3].values.astype(np.float32)

#log trans on dep var
y_log = np.log(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lower = 1e-5
upper = 1e5

kernel = C(1.0, (lower, upper)) * Matern(length_scale=[1.0, 1.0, 1.0],length_scale_bounds=(lower, upper),nu=2.5)
#kernel = C(1.0, (lower, upper)) * RBF([1.0, 1.0, 1.0], (lower, upper))

#train model
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=200,normalize_y=True,random_state=42)
gp.fit(X_train_scaled, y_train_log)

#predictions
X_new = np.array([[0.4, 1.0, 2.0]], dtype=np.float32)
X_new_scaled = scaler.transform(X_new)
y_pred_log, sigma_log = gp.predict(X_new_scaled, return_std=True)
y_pred = np.exp(y_pred_log)
#uncertanty
sigma = y_pred * sigma_log  # delta method

print(f"Predicted yield strength: {y_pred[0]:.2f} MPa")
print(f"Uncertainty: {sigma[0]:.2f} MPa")
print("Learned kernel:", gp.kernel_)

#test model usig test data
y_test_pred_log = gp.predict(X_test_scaled)
y_test_pred = np.exp(y_test_pred_log)
y_test_orig = np.exp(y_test_log)

#model metrics w/ test data
rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
r2 = r2_score(y_test_orig, y_test_pred)
test_mae = mean_absolute_error(y_test_orig, y_test_pred)

print(f"RMSE: {rmse:.2f} MPa")
print(f"R² Score: {r2:.3f}")
print(f"Test MAE: {test_mae:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_test_pred, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot([min(y_test_orig), max(y_test_orig)],[min(y_test_orig), max(y_test_orig)],color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Stress (MPa)")
plt.ylabel("Predicted Stress (MPa)")
plt.title("Actual vs. Predicted Stress Values")
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test_orig - y_test_pred
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


#NEW FOR HW 6
#-----------------------------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores_cv = []
rmse_scores_cv = []
mae_scores_cv = []

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    #data split
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv_log, y_test_cv_log = y_log[train_index], y_log[test_index]
    
    #std features
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    #build GPR model
    gp_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200, 
                                       normalize_y=True, random_state=42)
    gp_cv.fit(X_train_cv_scaled, y_train_cv_log)
    
    #predictions
    y_pred_cv_log, sigma_log_cv = gp_cv.predict(X_test_cv_scaled, return_std=True)
    
    #inv_log transfrom
    y_pred_cv = np.exp(y_pred_cv_log)
    y_test_cv_orig = np.exp(y_test_cv_log)
    
    #metrics
    r2 = r2_score(y_test_cv_orig, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y_test_cv_orig, y_pred_cv))
    mae = mean_absolute_error(y_test_cv_orig, y_pred_cv)
    
    r2_scores_cv.append(r2)
    rmse_scores_cv.append(rmse)
    mae_scores_cv.append(mae)
    
    print(f"Fold {fold} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

#metrics
avg_r2_cv = np.mean(r2_scores_cv)
avg_rmse_cv = np.mean(rmse_scores_cv)
avg_mae_cv = np.mean(mae_scores_cv)

print("\nAverage 5-Fold Cross Validation Metrics (Gaussian Process Regression):")
print(f"Average R²: {avg_r2_cv:.4f}")
print(f"Average RMSE: {avg_rmse_cv:.4f}")
print(f"Average MAE: {avg_mae_cv:.4f}")

#PLOT
folds = np.arange(1, 6)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(folds, rmse_scores_cv, marker='o', label='RMSE')
plt.plot(folds, mae_scores_cv, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (disp) (GPR)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, r2_scores_cv, marker='o', color='green', label='R²')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.title('R² per Fold (disp) (GPR)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
