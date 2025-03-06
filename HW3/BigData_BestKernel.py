import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score

#load data
file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 3\PropertySpace_csv.csv"
df = pd.read_csv(file_path)

df_partial = df.sample(n=1000, random_state=42)

#extract data
X = df_partial.iloc[:, :4].values #X (features): first 4 columns (stiffness tensor values [C11 C12 C22 C66])
y = df_partial.iloc[:, 4].values #y (target): 5th column (volume Fraction)

#set bounds, these result in the least non convergenses 
lower_const, upper_const = 1e-5, 1e5  
lower_rbf, upper_rbf = 1e-9, 1e9  
lower_matern, upper_matern = 1e-9, 1e9  
lower_white, upper_white = 1e-5, 1e-1  

#kernels to try
extended_kernels = {
    "Constant": C(1.0, (lower_const, upper_const)),
    "RBF": C(1.0, (lower_const, upper_const)) * RBF([1.0, 1.0, 1.0, 1.0], (lower_rbf, upper_rbf)),
    "RBF + Matern (ν=1.5)": C(1.0, (lower_const, upper_const)) * (RBF([1.0, 1.0, 1.0, 1.0], (lower_rbf, upper_rbf)) +Matern(length_scale=[1.0, 1.0, 1.0, 1.0], length_scale_bounds=(lower_matern, upper_matern),nu=1.5)),
    "RBF + Matern (ν=2.5)": C(1.0, (lower_const, upper_const)) * (RBF([1.0, 1.0, 1.0, 1.0], (lower_rbf, upper_rbf)) +Matern(length_scale=[1.0, 1.0, 1.0, 1.0],length_scale_bounds=(lower_matern, upper_matern),nu=2.5)),
    "RBF + Matern (ν=0.5)": C(1.0, (lower_const, upper_const)) * (RBF([1.0, 1.0, 1.0, 1.0], (lower_rbf, upper_rbf)) +Matern(length_scale=[1.0, 1.0, 1.0, 1.0],length_scale_bounds=(lower_matern, upper_matern),nu=0.5)),
    "Matern (ν=1.5)": C(1.0, (lower_const, upper_const)) * Matern(length_scale=[1.0, 1.0, 1.0, 1.0],length_scale_bounds=(lower_matern, upper_matern),nu=1.5),
    "Matern (ν=2.5)": C(1.0, (lower_const, upper_const)) * Matern(length_scale=[1.0, 1.0, 1.0, 1.0],length_scale_bounds=(lower_matern, upper_matern),nu=2.5),
    "RBF + White Noise": C(1.0, (lower_const, upper_const)) * RBF([1.0, 1.0, 1.0, 1.0], (lower_rbf, upper_rbf)) +WhiteKernel(noise_level=1e-3, noise_level_bounds=(lower_white, upper_white)),
    "Matern (ν=1.5) + White Noise": C(1.0, (lower_const, upper_const)) * Matern(length_scale=[1.0, 1.0, 1.0, 1.0],length_scale_bounds=(lower_matern, upper_matern),nu=1.5) +WhiteKernel(noise_level=1e-3, noise_level_bounds=(lower_white, upper_white)),
}

#k fold cv
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {name: {"rmse": [], "r2": []} for name in extended_kernels}

#do CV
for train_index, test_index in kf.split(X):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    scaler = StandardScaler()
    X_train_scaled_cv = scaler.fit_transform(X_train_cv)
    X_test_scaled_cv = scaler.transform(X_test_cv)
    
    for name, kernel in extended_kernels.items():
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200,alpha=1e-6, normalize_y=True, random_state=42)
        gp.fit(X_train_scaled_cv, y_train_cv)
        y_test_pred_cv = gp.predict(X_test_scaled_cv)
        
        y_test_pred_orig = np.exp(y_test_pred_cv)
        y_test_orig = np.exp(y_test_cv)
        
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
        r2 = r2_score(y_test_orig, y_test_pred_orig)
        results[name]["rmse"].append(rmse)
        results[name]["r2"].append(r2)

#metrics
avg_results = {}
for name, metrics in results.items():
    avg_rmse = np.mean(metrics["rmse"])
    avg_r2 = np.mean(metrics["r2"])
    avg_results[name] = {"avg_rmse": avg_rmse, "avg_r2": avg_r2}

#sort by best RMSE and R^2
sorted_by_rmse = sorted(avg_results.items(), key=lambda x: x[1]["avg_rmse"])
sorted_by_r2 = sorted(avg_results.items(), key=lambda x: x[1]["avg_r2"], reverse=True)

print("Top 3 Kernels by average RMSE:")
for i, (name, metrics) in enumerate(sorted_by_rmse[:3], start=1):
    print(f"{i}. {name}: avg RMSE = {metrics['avg_rmse']:.4f}")

print("\nTop 3 Kernels by average R²:")
for i, (name, metrics) in enumerate(sorted_by_r2[:3], start=1):
    print(f"{i}. {name}: avg R² = {metrics['avg_r2']:.4f}")
