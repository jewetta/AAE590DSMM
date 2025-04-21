import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

#OLD FROM HW 2

#load data
#file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 2\PropertySpace_csv.csv"
file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 2/PropertySpace_csv.csv'
df = pd.read_csv(file_path)

#X = df.iloc[:, :4].values #X (features): first 4 columns (stiffness tensor values [C11 C12 C22 C66])
#y = df.iloc[:, 4].values #y (target): 5th column (volume Fraction)

X = df.iloc[:, [1, 2, 3, 4]].values  # X (features): columns 0, 2, 3, and 4
y = df.iloc[:, 0].values             # y (target): column 1


#split the data w/ train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alpha = 0.0001 #found opt value by min MSE in other script
lasso = Lasso(alpha=alpha, random_state=42)
lasso.fit(X_train_scaled, y_train)

alpha = 0.151991 #found opt value by min MSE in other script
ridge = Ridge(alpha=alpha, random_state=42)  
ridge.fit(X_train_scaled, y_train)

#predictions
y_predL = lasso.predict(X_test_scaled)
y_predR = ridge.predict(X_test_scaled)

#model metrics
mseL = mean_squared_error(y_test, y_predL)
r2L = r2_score(y_test, y_predL)
mseR = mean_squared_error(y_test, y_predR)
r2R = r2_score(y_test, y_predR)

print("Lasso Model:")
print(f"Mean Squared Error: {mseL:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseL):.4f}")
print(f"R² Score: {r2L:.4f}")

print("\nRidge Model:")
print(f"Mean Squared Error: {mseR:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseR):.4f}")
print(f"R² Score: {r2R:.4f}")

#;asso coefficients
feature_names = ['C11', 'C12', 'C22', 'C66']
print("\nLasso Coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    print(f"{name}: {coef:.4f}")

#ridge coefficients
feature_names = ['C11', 'C12', 'C22', 'C66']
print("\nRidge Coefficients:")
for name, coef in zip(feature_names, ridge.coef_):
    print(f"{name}: {coef:.4f}")

#predictions funts
def predict_volume_fractionL(X_input):
    X_array = np.array(X_input).reshape(1, -1)  
    X_scaled = scaler.transform(X_array)  
    y_pred = lasso.predict(X_scaled)  
    return y_pred[0]

def predict_volume_fractionR(X_input):
    X_array = np.array(X_input).reshape(1, -1)  
    X_scaled = scaler.transform(X_array)  
    y_pred = ridge.predict(X_scaled)
    return y_pred[0]

#test prediction
example_input = [0.1, 0.5, 0.6, 0.3]  # <-- input stiffness tensor values here
predicted_volume_fractionL = predict_volume_fractionL(example_input)
predicted_volume_fractionR = predict_volume_fractionR(example_input)
print(f"\nLasso Predicted Volume Fraction for input {example_input}: {predicted_volume_fractionL:.4f}")
print(f"Ridge Predicted Volume Fraction for input {example_input}: {predicted_volume_fractionR:.4f}")

#plots

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predL, color='blue', alpha=0.01, label='Data Points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Volume Fraction')
plt.ylabel('Predicted Volume Fraction')
plt.title('Lasso: Actual vs Predicted Volume Fraction')
plt.legend()
plt.show(block=False)

plt.figure(figsize=(8,6))
residualsL = y_test - y_predL 
plt.scatter(y_predL, residualsL, color='green', alpha=0.01)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Predicted Volume Fraction')
plt.ylabel('Residuals')
plt.title('Lasso Residuals vs Predicted Volume Fraction')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predR, color='blue', alpha=0.01, label='Data Points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Volume Fraction')
plt.ylabel('Predicted Volume Fraction')
plt.title('Ridge: Actual vs Predicted Volume Fraction')
plt.legend()
plt.show(block=False)


plt.figure(figsize=(8,6))
residualsR = y_test - y_predR  
plt.scatter(y_predR, residualsR, color='green', alpha=0.01)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Predicted Volume Fraction')
plt.ylabel('Residuals')
plt.title('Ridge Residuals vs Predicted Volume Fraction')
plt.show()

coefficients = lasso.coef_
plt.figure(figsize=(8,6))
plt.bar(feature_names, coefficients, color='purple')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.show()

plt.figure(figsize=(8,6))
plt.bar(feature_names, ridge.coef_, color='orange')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients')
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(residualsR, dist="norm", plot=plt)
plt.xlabel("Theoretical quantiles")  
plt.ylabel("Ordered values") 
plt.title("Probability Plot for Ridge Regression")
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(residualsL, dist="norm", plot=plt)
plt.xlabel("Theoretical quantiles")  
plt.ylabel("Ordered values") 
plt.title("Probability Plot for Lasso Regression")
plt.show()


#NEW FOR HW 6
#---------------------------------------------

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


#alpha values (defined in hw 2)
alpha_lasso = 0.0001
alpha_ridge = 0.151991

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lasso_r2_scores = []
lasso_adj_r2_scores = []
lasso_rmse_scores = []
lasso_mae_scores = []

print("\n5-Fold Cross Validation Metrics (Lasso Model):")

#for lasso model

#implement 5-fold CV
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
    y_train_cv, y_test_cv = y[train_idx], y[test_idx]
    
    #std indep vars
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    #build models
    model_lasso_cv = Lasso(alpha=alpha_lasso, random_state=42)
    model_lasso_cv.fit(X_train_cv_scaled, y_train_cv)
    
    #predictions
    y_pred_cv = model_lasso_cv.predict(X_test_cv_scaled)
    
    #metrics
    r2_cv = r2_score(y_test_cv, y_pred_cv)
    mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
    rmse_cv = np.sqrt(mse_cv)
    mae_cv = mean_absolute_error(y_test_cv, y_pred_cv)
    
    #adjusted R² calculation: adjusted R² = 1 - (1-R²)*(n-1)/(n-p-1)
    n_samples = len(y_test_cv)
    n_features = X_test_cv.shape[1]
    adj_r2_cv = 1 - (1 - r2_cv) * (n_samples - 1) / (n_samples - n_features - 1)
    
    lasso_r2_scores.append(r2_cv)
    lasso_adj_r2_scores.append(adj_r2_cv)
    lasso_rmse_scores.append(rmse_cv)
    lasso_mae_scores.append(mae_cv)
    
    print(f"Fold {fold} (Lasso) - R²: {r2_cv:.4f}, Adjusted R²: {adj_r2_cv:.4f}, RMSE: {rmse_cv:.4f}, MAE: {mae_cv:.4f}")

#average metrics across 5 folds
avg_r2_lasso = np.mean(lasso_r2_scores)
avg_adj_r2_lasso = np.mean(lasso_adj_r2_scores)
avg_rmse_lasso = np.mean(lasso_rmse_scores)
avg_mae_lasso = np.mean(lasso_mae_scores)

print("\nAverage 5-Fold CV Metrics (Lasso Model):")
print(f"Average R²: {avg_r2_lasso:.4f}")
print(f"Average Adjusted R²: {avg_adj_r2_lasso:.4f}")
print(f"Average RMSE: {avg_rmse_lasso:.4f}")
print(f"Average MAE: {avg_mae_lasso:.4f}")

#plots
folds = np.arange(1, 6)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(folds, lasso_rmse_scores, marker='o', label='RMSE')
plt.plot(folds, lasso_mae_scores, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (Lasso)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, lasso_r2_scores, marker='o', color='green', label='R²')
plt.plot(folds, lasso_adj_r2_scores, marker='o', color='blue', label='Adjusted R²')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('R² & Adjusted R² per Fold (Lasso)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#now same for ridge model

kf = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_r2_scores = []
ridge_adj_r2_scores = []
ridge_rmse_scores = []
ridge_mae_scores = []

print("\n5-Fold Cross Validation Metrics (Ridge Model):")
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
    y_train_cv, y_test_cv = y[train_idx], y[test_idx]
    
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    model_ridge_cv = Ridge(alpha=alpha_ridge, random_state=42)
    model_ridge_cv.fit(X_train_cv_scaled, y_train_cv)
    
    y_pred_cv = model_ridge_cv.predict(X_test_cv_scaled)
    
    r2_cv = r2_score(y_test_cv, y_pred_cv)
    mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
    rmse_cv = np.sqrt(mse_cv)
    mae_cv = mean_absolute_error(y_test_cv, y_pred_cv)
    
    n_samples = len(y_test_cv)
    n_features = X_test_cv.shape[1]
    adj_r2_cv = 1 - (1 - r2_cv) * (n_samples - 1) / (n_samples - n_features - 1)
    
    ridge_r2_scores.append(r2_cv)
    ridge_adj_r2_scores.append(adj_r2_cv)
    ridge_rmse_scores.append(rmse_cv)
    ridge_mae_scores.append(mae_cv)
    
    print(f"Fold {fold} (Ridge) - R²: {r2_cv:.4f}, Adjusted R²: {adj_r2_cv:.4f}, RMSE: {rmse_cv:.4f}, MAE: {mae_cv:.4f}")

avg_r2_ridge = np.mean(ridge_r2_scores)
avg_adj_r2_ridge = np.mean(ridge_adj_r2_scores)
avg_rmse_ridge = np.mean(ridge_rmse_scores)
avg_mae_ridge = np.mean(ridge_mae_scores)

print("\nAverage 5-Fold CV Metrics (Ridge Model):")
print(f"Average R²: {avg_r2_ridge:.4f}")
print(f"Average Adjusted R²: {avg_adj_r2_ridge:.4f}")
print(f"Average RMSE: {avg_rmse_ridge:.4f}")
print(f"Average MAE: {avg_mae_ridge:.4f}")

#ridge plots
folds = np.arange(1, 6)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(folds, ridge_rmse_scores, marker='o', label='RMSE')
plt.plot(folds, ridge_mae_scores, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (Ridge)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, ridge_r2_scores, marker='o', color='green', label='R²')
plt.plot(folds, ridge_adj_r2_scores, marker='o', color='blue', label='Adjusted R²')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('R² & Adjusted R² per Fold (Ridge)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
