import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats
from sklearn.model_selection import KFold


#OLD FROM HW 3

#file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 3\PropertySpace_csv.csv"
file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 3/PropertySpace_csv.csv'
df = pd.read_csv(file_path)

df_partial = df.sample(n=5000, random_state=49)

#X = df_partial.iloc[:, :4].values #X (features): first 4 columns (stiffness tensor values [C11 C12 C22 C66])
#y = df_partial.iloc[:, 4].values #y (target): 5th column (volume Fraction)

X = df_partial.iloc[:, [1, 2, 3, 4]].values  # X (features): columns 0, 2, 3, and 4
y = df_partial.iloc[:, 0].values             # y (target): column 1

#try for comp w homework 5
#X = df.iloc[:, 2:3].values  # shape: (num_samples, 5)
#y = df.iloc[:, 1].values.reshape(-1, 1)

#split the data w/ train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#std the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#.astype(np.float32)


lower=1e-5
upper=1e5
#define kernel from best check
kernel = C(1.0, (lower, upper)) * RBF([1.0, 1.0, 1.0, 1.0], (lower, upper)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(lower, upper)) #optimized w/ optimizer

# #do GPR
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200, normalize_y=True, random_state=42)
# gp.fit(X_train_scaled, y_train)

# #predict for a new composition
# X_new = np.array([[0.4, 0.6, 0.2, 0.5]])#, dtype=np.float32)
# X_new_scaled = scaler.transform(X_new)
# y_pred, sigma = gp.predict(X_new_scaled, return_std=True)

# print(f"Predicted VF: {y_pred[0]:.2f}")
# print(f"Uncertainty: {sigma[0]:.2f}")
# print("Learned kernel:", gp.kernel_)

# #predict on the test set
# y_test_pred = gp.predict(X_test_scaled)

# #metrics
# rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# r2 = r2_score(y_test, y_test_pred)
# test_mae = mean_absolute_error(y_test, y_test_pred)
# mse =mean_squared_error(y_test, y_test_pred)

# print(f"RMSE: {rmse:.6f}")
# print(f"Test MSE: {mse:.6f}")
# print(f"R² Score: {r2:.4f}")
# print(f"Test MAE: {test_mae:.6f}")


# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_test_pred, color="blue", alpha=0.7, label="Predicted vs Actual")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
# plt.xlabel("Actual VF")
# plt.ylabel("Predicted VF")
# plt.title("Actual vs. Predicted VF Values")
# plt.legend()
# plt.grid(True)
# plt.show()

# residuals = y_test - y_test_pred

# plt.figure(figsize=(8, 6))
# plt.scatter(y_test_pred, residuals, color="blue", alpha=0.6)
# plt.axhline(y=0, color="red", linestyle="--")
# plt.xlabel("Predicted VF")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.grid(True)
# plt.show()

#NEW FOR HW 6
#-----------------------------------------

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rmse_list = []
cv_r2_list = []
cv_mae_list = []
cv_adj_r2_list = []

#implemnt 5-fold CV
for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
    #test/train data split per fold
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    #std indep vars
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    #build model
    gp_cv = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42
    )
    gp_cv.fit(X_train_cv_scaled, y_train_cv)
    
    #preductions
    y_pred_cv = gp_cv.predict(X_test_cv_scaled)
    
    #metrics per fold
    fold_rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
    fold_r2 = r2_score(y_test_cv, y_pred_cv)
    fold_mae = mean_absolute_error(y_test_cv, y_pred_cv)
    
    #adjusted R²: adjusted R² = 1 - (1-R²)*((n-1)/(n-p-1))
    n_samples = len(y_test_cv)
    n_features = X_test_cv.shape[1]
    fold_adj_r2 = 1 - (1 - fold_r2) * (n_samples - 1) / (n_samples - n_features - 1)
    
    cv_rmse_list.append(fold_rmse)
    cv_r2_list.append(fold_r2)
    cv_mae_list.append(fold_mae)
    cv_adj_r2_list.append(fold_adj_r2)
    
    print(f"Fold {fold_idx} - R²: {fold_r2:.4f}, Adjusted R²: {fold_adj_r2:.4f}, RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}")

#average metrics over 5 folds
avg_rmse = np.mean(cv_rmse_list)
avg_r2 = np.mean(cv_r2_list)
avg_adj_r2 = np.mean(cv_adj_r2_list)
avg_mae = np.mean(cv_mae_list)

print("\n5-Fold Cross-Validation Results:")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average R²: {avg_r2:.4f}")
print(f"Average Adjusted R²: {avg_adj_r2:.4f}")
print(f"Average MAE: {avg_mae:.4f}")


folds = np.arange(1, 6)
#PLOTS
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(folds, cv_rmse_list, marker='o', label='RMSE')
plt.plot(folds, cv_mae_list, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (GPR)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, cv_r2_list, marker='o', color='green', label='R²')
plt.plot(folds, cv_adj_r2_list, marker='o', color='blue', label='Adjusted R²')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('R² & Adjusted R² per Fold (GPR)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
