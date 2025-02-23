import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

#load dataset - note I deleted largest outlier (35mm def) to better results
file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 2\DataSet_Disp_Import_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :3].values #columns 1, 2, 3 -> Features [VF, displacement, material]
y = df.iloc[:, 3].values #column 4 -> target (max disp (mm))

#split the data w/ train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#make models
alpha = 0.1 
lasso = Lasso(alpha=alpha, random_state=42)
lasso.fit(X_train_scaled, y_train)

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

print("\nRidge Model")
print(f"Mean Squared Error: {mseR:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseR):.4f}")
print(f"R² Score: {r2R:.4f}")


#lasso coefficients
feature_names = ['X1 (Volume Fraction)', 'X2 (Load (N))', 'X3 (Material)']
print("\nLasso Coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    print(f"{name}: {coef:.4f}")

#prediction functions
def predict_dispL(X_input):
    X_array = np.array(X_input).reshape(1, -1)  
    X_scaled = scaler.transform(X_array)  
    y_pred = lasso.predict(X_scaled)  
    return y_pred[0]
def predict_dispR(X_input):
    X_array = np.array(X_input).reshape(1, -1)  
    X_scaled = scaler.transform(X_array)  
    y_pred = ridge.predict(X_scaled)  
    return y_pred[0]

#test predicton
example_input = [0.2, 3, 1]  # <-- values here [Volume fract, load(in hundreds of newtons), material(1,2,3)]
predicted_dispL = predict_dispL(example_input)
print(f"\nLasso Predicted Disp for input {example_input}: {predicted_dispL:.2f}")
predicted_dispR = predict_dispR(example_input)
print(f"Ridge Predicted Disp for input {example_input}: {predicted_dispR:.2f}")


#plots

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predL, color='blue', label='Data Points')  
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Fit')  
plt.xlabel('Actual Disp')
plt.ylabel('Predicted Disp')
plt.title('Lasso Actual vs Predicted Disp Values')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predR, color='blue', label='Data Points')  
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Fit')  
plt.xlabel('Actual Disp')
plt.ylabel('Predicted Disp')
plt.title('Ridge Actual vs Predicted Disp Values')
plt.legend()
plt.show()

residualsL = y_test - y_predL 
plt.figure(figsize=(8,6))
plt.scatter(y_predL, residualsL, color='green', alpha=0.6) 
plt.axhline(0, color='red', linewidth=2) 
plt.xlabel('Predicted Disp')
plt.ylabel('Residuals')
plt.title('Lasso Residuals vs Predicted Disp')
plt.show()

residualsR = y_test - y_predR  
plt.figure(figsize=(8,6))
plt.scatter(y_predR, residualsR, color='green', alpha=0.6)  
plt.axhline(0, color='red', linewidth=2) 
plt.xlabel('Predicted Disp')
plt.ylabel('Residuals')
plt.title('Ridge Residuals vs Predicted Disp')
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