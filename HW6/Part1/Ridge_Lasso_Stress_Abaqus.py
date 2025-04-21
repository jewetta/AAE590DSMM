import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

#OLD FROM HW 2

#data load
file_path = r"/Users/austinjewett/Desktop/Purdue/AAE590/HW 2/DataSet_Stress_Import_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :3].values #columns 1, 2, 3 -> Features [VF, displacement, material]
y = df.iloc[:, 3].values #column 4 -> target (max stress)

# Perform log transform on target variable (y)
y = np.log(y)

#split the data w/ train set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#build models
alpha = 0.1  
lasso = Lasso(alpha=alpha, random_state=42)
lasso.fit(X_train_scaled, y_train)

ridge = Ridge(alpha=alpha, random_state=42)  
ridge.fit(X_train_scaled, y_train)

print(f"\nLasso Model Y-Intercept: {lasso.intercept_:.4f}")
print(f"Ridge Model Y-Intercept: {ridge.intercept_:.4f}")

#predictions (predictions are in log scale)
y_predL = lasso.predict(X_test_scaled)
y_predR = ridge.predict(X_test_scaled)

# Apply inverse log transform to obtain predictions and targets in original scale
y_predL_orig = np.exp(y_predL)
y_predR_orig = np.exp(y_predR)
y_test_orig = np.exp(y_test)

#model metrics computed on original scale
mseL = mean_squared_error(y_test_orig, y_predL_orig)
r2L = r2_score(y_test_orig, y_predL_orig)
mseR = mean_squared_error(y_test_orig, y_predR_orig)
r2R = r2_score(y_test_orig, y_predR_orig)

print("Lasso Model:")
print(f"Mean Squared Error: {mseL:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseL):.4f}")
print(f"R² Score: {r2L:.4f}")

print("\nRidge Model")
print(f"Mean Squared Error: {mseR:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseR):.4f}")
print(f"R² Score: {r2R:.4f}")

#print coefficients
feature_names = ['X1 (Volume Fraction)', 'X2 (Disp)', 'X3 (Material)']
print("\nLasso Coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    print(f"{name}: {coef:.4f}")

print("\nRidge Coefficients:")
for name, coef in zip(feature_names, ridge.coef_):
    print(f"{name}: {coef:.4f}")

#prediction functs
def predict_stressL(X_input):
    X_array = np.array(X_input).reshape(1, -1)  
    X_scaled = scaler.transform(X_array)  
    y_pred = lasso.predict(X_scaled)  
    return np.exp(y_pred[0])  # Inverse log transform before returning

def predict_stressR(X_input):
    X_array = np.array(X_input).reshape(1, -1)  
    X_scaled = scaler.transform(X_array) 
    y_pred = ridge.predict(X_scaled) 
    return np.exp(y_pred[0])  # Inverse log transform before returning

#prediction out 
example_input = [0.2, 1, 1]  # <-- test inpits [Volume fraction, Displacment(mm), materaiL(1,2,3)]
predicted_stressL = predict_stressL(example_input)
print(f"\nLasso Predicted Stress for input {example_input}: {predicted_stressL:.2f}")
predicted_stressR = predict_stressR(example_input)
print(f"Ridge Predicted Stress for input {example_input}: {predicted_stressR:.2f}")

#plots

plt.figure(figsize=(8,6))
plt.scatter(y_test_orig, y_predL_orig, color='blue', label='Data Points')  
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], color='red', linewidth=2, label='Ideal Fit')  
plt.xlabel('Actual Stress')
plt.ylabel('Predicted Stress')
plt.title('Lasso Actual vs Predicted Stress Values')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test_orig, y_predR_orig, color='blue', label='Data Points') 
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], color='red', linewidth=2, label='Ideal Fit')  
plt.xlabel('Actual Stress')
plt.ylabel('Predicted Stress')
plt.title('Ridge Actual vs Predicted Stress Values')
plt.legend()
plt.show()

residualsL = y_test_orig - y_predL_orig  
plt.figure(figsize=(8,6))
plt.scatter(y_predL_orig, residualsL, color='green', alpha=0.6)  
plt.axhline(0, color='red', linewidth=2) 
plt.xlabel('Predicted Stress')
plt.ylabel('Residuals')
plt.title('Lasso Residuals vs Predicted Stress')
plt.show()

residualsR = y_test_orig - y_predR_orig 
plt.figure(figsize=(8,6))
plt.scatter(y_predR_orig, residualsR, color='green', alpha=0.6) 
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Predicted Stress')
plt.ylabel('Residuals')
plt.title('Ridge Residuals vs Predicted Stress')
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
plt.ylabel("Sample values") 
plt.title("Normal Probability plot of residuals for Ridge Regression")
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(residualsL, dist="norm", plot=plt)
plt.xlabel("Theoretical quantiles")  
plt.ylabel("Sample values") 
plt.title("Normal Probability plot of residuals for Lasso Regression")
plt.show()


#NEW FOR HW 6
#-------------------------------------

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
rmse_scores = []
mae_scores = []

#implement 5-fold CV
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    #std featrues
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    #built model
    model_cv = Lasso(alpha=alpha, random_state=42)
    model_cv.fit(X_train_cv_scaled, y_train_cv)
    
    #predictions
    y_pred_cv = model_cv.predict(X_test_cv_scaled)
    
    #inv_log transfrom
    y_pred_cv_orig = np.exp(y_pred_cv)
    y_test_cv_orig = np.exp(y_test_cv)
    
    #metrics
    r2 = r2_score(y_test_cv_orig, y_pred_cv_orig)
    mse_cv = mean_squared_error(y_test_cv_orig, y_pred_cv_orig)
    rmse = np.sqrt(mse_cv)
    mae = mean_absolute_error(y_test_cv_orig, y_pred_cv_orig)
    
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    
    print(f"Fold {fold} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

#average metrics over all 5 folds
avg_r2 = sum(r2_scores) / 5
avg_rmse = sum(rmse_scores) / 5
avg_mae = sum(mae_scores) / 5

print("\nAverage 5-Fold Cross Validation Metrics (Lasso Model):")
print(f"Average R²: {avg_r2:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")

#PLOTS
folds = np.arange(1, 6)  

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.plot(folds, rmse_scores, marker='o', label='RMSE')
plt.plot(folds, mae_scores, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (stress) (Lasso)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, r2_scores, marker='o', color='green', label='R²')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.title('R² per Fold (stress) (Lasso)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#Now for ridge

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores_ridge = []
rmse_scores_ridge = []
mae_scores_ridge = []

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    #std features
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
    
    #build model
    model_cv = Ridge(alpha=alpha, random_state=42)
    model_cv.fit(X_train_cv_scaled, y_train_cv)
    
    #predict
    y_pred_cv = model_cv.predict(X_test_cv_scaled)
    
    #inv_log transform
    y_pred_cv_orig = np.exp(y_pred_cv)
    y_test_cv_orig = np.exp(y_test_cv)
    
    #metic calcs
    r2 = r2_score(y_test_cv_orig, y_pred_cv_orig)
    mse_cv = mean_squared_error(y_test_cv_orig, y_pred_cv_orig)
    rmse = np.sqrt(mse_cv)
    mae = mean_absolute_error(y_test_cv_orig, y_pred_cv_orig)
    
    r2_scores_ridge.append(r2)
    rmse_scores_ridge.append(rmse)
    mae_scores_ridge.append(mae)
    
    print(f"Fold {fold} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

#average metrics over all 5 folds
avg_r2_ridge = sum(r2_scores_ridge) / 5
avg_rmse_ridge = sum(rmse_scores_ridge) / 5
avg_mae_ridge = sum(mae_scores_ridge) / 5

print("\nAverage 5-Fold Cross Validation Metrics (Ridge Model):")
print(f"Average R²: {avg_r2_ridge:.4f}")
print(f"Average RMSE: {avg_rmse_ridge:.4f}")
print(f"Average MAE: {avg_mae_ridge:.4f}")

#PLOTS
folds = np.arange(1, 6)  

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.plot(folds, rmse_scores_ridge, marker='o', label='RMSE')
plt.plot(folds, mae_scores_ridge, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('RMSE & MAE per Fold (stress) (Ridge)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, r2_scores_ridge, marker='o', color='green', label='R²')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.title('R² per Fold (stress) (Ridge)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
