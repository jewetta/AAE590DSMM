import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import arviz as az
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import scipy.stats as stats


#load data
file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 3\DataSet_Stress_Import_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :3].values.astype(np.float32)
y = df.iloc[:, 3].values.astype(np.float32)

#dep var transform
y_log = np.log(y)

# Histogram of y_log using Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(y_log, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Log-transformed Dependent Variable (y_log)")
plt.ylabel("Frequency")
plt.title("Histogram of Log-transformed Dependent Variable")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

#standardize indep vars
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

#build model
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=6, sigma=3)  
    betas = pm.Uniform('betas', lower=-1, upper=2, shape=X_train_scaled.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + pm.math.dot(X_train_scaled, betas)

    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train_log)

    trace = pm.sample(2000, cores=1, tune=1000, return_inferencedata=True)

print(az.summary(trace))
az.plot_trace(trace)
plt.show()

ppc = pm.sample_posterior_predictive(trace, model=model)
az.plot_ppc(ppc, figsize=(10, 5))
plt.show()

az.plot_posterior(trace, var_names=['alpha', 'betas', 'sigma'], textsize=14, point_estimate='mean', rope_color='black')
plt.show()

#validate model w/ test set
posterior_samples = trace.posterior
alpha_samples = posterior_samples["alpha"].values.flatten()  
beta_samples = posterior_samples["betas"].values.reshape(-1, X_test.shape[1])

#make predictions w/ model for test set
test_preds_log = []
for i in range(len(alpha_samples)):
    mu_i = alpha_samples[i] + np.dot(X_test_scaled, beta_samples[i])
    test_preds_log.append(mu_i)
test_preds_log = np.array(test_preds_log)  

#back to normal
test_preds = np.exp(test_preds_log)
test_mean_preds = np.mean(test_preds, axis=0)
y_test_orig = np.exp(y_test_log)

#metrics
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_mean_preds))
test_r2 = r2_score(y_test_orig, test_mean_preds)
test_mae = mean_absolute_error(y_test_orig, test_mean_preds)

print("\nModel Performance on TEST data")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R^2 : {test_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")

#act vs predicted
plt.figure()
plt.scatter(y_test_orig, test_mean_preds, alpha=0.7)
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], linestyle='--', color='gray', label='Perfect Prediction')
plt.xlabel("Observed Stress (Test)")
plt.ylabel("Posterior Mean Predicted Stress")
plt.title("Test Set: Observed vs. Predicted")
plt.legend()
plt.show()

#residuals
residuals = y_test_orig - test_mean_preds
plt.figure(figsize=(8, 6))
plt.scatter(test_mean_preds, residuals, alpha=0.7)
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Predicted Stress")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

sns.histplot(residuals, kde=True)
plt.axvline(0, linestyle="--", color="gray")
plt.xlabel("Residuals")
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(8,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()