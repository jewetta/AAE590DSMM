import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import arviz as az
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats

#OLD STUFF FROM HW 3

#load data
file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 3/PropertySpace_csv.csv'
#file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/PropertySpace.csv'
df = pd.read_csv(file_path)

df_partial = df.sample(n=100000, random_state=42) #change based on computer (larger data sizes take longer to run)

#X = df_partial.iloc[:, 1:5].values #X (features): first 4 columns (stiffness tensor values [C11 C12 C22 C66])
#y = df_partial.iloc[:, 0].values #y (target): 5th column (volume Fraction)

X = df_partial.iloc[:, [1, 2, 3, 4]].values  # X (features): columns 0, 2, 3, and 4
y = df_partial.iloc[:, 0].values             # y (target): column 1

#X = df.iloc[:, 2:3].values  # shape: (num_samples, 5)
#y = df.iloc[:, 1].values.reshape(-1, 1)

#split the data w/ train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# #train model
# with pm.Model() as model:
#   #  alpha = pm.Normal('alpha', mu=0, sigma=25)      # intercept
#   #  betas = pm.Normal('betas', mu=0, sigma=25, shape=X_train_scaled.shape[1])  # slopes
#   #  sigma = pm.HalfNormal('sigma', sigma=1)

#     alpha = pm.Normal('alpha', mu=0.7, sigma=1)  
#     #betas = pm.StudentT('betas', nu=3, mu=0.3, sigma=2, shape=X_train_scaled.shape[1])  
#     betas = pm.Uniform('betas', lower=-3, upper=3, shape=X_train_scaled.shape[1])
#    # betas = pm.TruncatedNormal('betas', mu=0, sigma=0.5, lower=-0.2, upper=0.2, shape=X_train_scaled.shape[1])
#     sigma = pm.HalfNormal('sigma', sigma=0.5)  

#     mu = alpha + pm.math.dot(X_train_scaled, betas)

#     Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train)

#     trace = pm.sample(1000, tune=5000, return_inferencedata=True) #2000, 1000

# print(az.summary(trace))
# az.plot_trace(trace)
# plt.show()

# ppc = pm.sample_posterior_predictive(trace, model=model)

# az.plot_ppc(ppc, figsize=(10, 5))
# plt.show()

# az.plot_posterior(trace,var_names=['alpha', 'betas', 'sigma'], textsize=14, point_estimate='mean',rope_color='black')
# plt.show()

# #validate on test set
# posterior_samples = trace.posterior
# alpha_samples = posterior_samples["alpha"].values.flatten() 
# beta_samples = posterior_samples["betas"].values.reshape(-1, X_test.shape[1])  

# test_preds = []
# for i in range(len(alpha_samples)):
#     mu_i = alpha_samples[i] + np.dot(X_test_scaled, beta_samples[i])
#     test_preds.append(mu_i)  

# test_preds = np.array(test_preds) 

# # take mean to be estimate
# test_mean_preds = np.mean(test_preds, axis=0)

# #Compute model metrics on test set
# test_rmse = np.sqrt(mean_squared_error(y_test, test_mean_preds))
# test_r2 = r2_score(y_test, test_mean_preds)
# test_mae = mean_absolute_error(y_test, test_mean_preds)

# print("\n--- Model Performance on TEST data ---")
# print(f"Test RMSE: {test_rmse:.4f}")
# print(f"Test R^2 : {test_r2:.4f}")
# print(f"Test MAE: {test_mae:.4f}")

# #plot act vs pred
# plt.figure()
# plt.scatter(y_test, test_mean_preds, alpha=0.05)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray', label='Perfect Prediction')
# plt.xlabel("Observed VF (Test)")
# plt.ylabel("Posterior Mean Predicted VF")
# plt.title("Test Set: Observed vs. Predicted")
# plt.legend()
# plt.show()

# #plot residuals
# residuals = y_test - test_mean_preds

# plt.figure(figsize=(8, 6))
# plt.scatter(test_mean_preds, residuals, alpha=0.05)
# plt.axhline(0, linestyle="--", color="gray")
# plt.xlabel("Predicted VF")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.show()

# plt.figure(figsize=(8, 6))
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.grid(True)
# plt.show()


#NEW FOR HW6

from sklearn.model_selection import KFold
import pymc as pm
import arviz as az  
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


kf = KFold(n_splits=5, shuffle=True, random_state=42)


r2_scores = []
adj_r2_scores = []
rmse_scores = []
mae_scores = []

print("\n--- 5-Fold Cross Validation Metrics (PyMC Bayesian Model) ---")
fold_num = 1
#implemnt 5-fold CV
for train_index, test_index in kf.split(X):
    #split to test/train per fold
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    #std indep vars
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv).astype(np.float32)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv).astype(np.float32)
    
    #built model (params set in HW 3)
    with pm.Model() as cv_model:
        alpha = pm.Normal('alpha', mu=0.7, sigma=1)
        betas = pm.Uniform('betas', lower=-3, upper=3, shape=X_train_cv_scaled.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=0.5)
        
        mu = alpha + pm.math.dot(X_train_cv_scaled, betas)
        
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train_cv)
        
        cv_trace = pm.sample(1000, tune=5000, return_inferencedata=True, progressbar=True)
    
    posterior_samples = cv_trace.posterior
    alpha_samples = posterior_samples["alpha"].values.flatten()
    beta_samples = posterior_samples["betas"].values.reshape(-1, X_test_cv_scaled.shape[1])

    test_preds = []
    for i in range(len(alpha_samples)):
        mu_i = alpha_samples[i] + np.dot(X_test_cv_scaled, beta_samples[i])
        test_preds.append(mu_i)
    test_preds = np.array(test_preds)

    test_mean_preds = np.mean(test_preds, axis=0)
    
    #metrics
    r2_cv = r2_score(y_test_cv, test_mean_preds)
    mse_cv = mean_squared_error(y_test_cv, test_mean_preds)
    rmse_cv = np.sqrt(mse_cv)
    mae_cv = mean_absolute_error(y_test_cv, test_mean_preds)
    
    #adjusted R²: Adjusted R² = 1 - (1-R²)*(n-1)/(n-p-1)
    n_samples = len(y_test_cv)
    n_features = X_test_cv_scaled.shape[1]
    adj_r2_cv = 1 - (1 - r2_cv) * (n_samples - 1) / (n_samples - n_features - 1)

    r2_scores.append(r2_cv)
    adj_r2_scores.append(adj_r2_cv)
    rmse_scores.append(rmse_cv)
    mae_scores.append(mae_cv)
    
    #metrics for this fold
    print(f"Fold {fold_num} - R²: {r2_cv:.4f}, Adjusted R²: {adj_r2_cv:.4f}, RMSE: {rmse_cv:.4f}, MAE: {mae_cv:.4f}")
    fold_num += 1

#average memtrics over 5 folds
avg_r2 = np.mean(r2_scores)
avg_adj_r2 = np.mean(adj_r2_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)

print("\n--- Average 5-Fold CV Metrics (PyMC Bayesian Model) ---")
print(f"Average R²: {avg_r2:.4f}")
print(f"Average Adjusted R²: {avg_adj_r2:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")

#pLots
folds = np.arange(1, 6)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(folds, rmse_scores, marker='o', label='RMSE')
plt.plot(folds, mae_scores, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (BLR)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, r2_scores, marker='o', color='green', label='R²')
plt.plot(folds, adj_r2_scores, marker='o', color='blue', label='Adjusted R²')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('R² & Adjusted R² per Fold (BLR)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
