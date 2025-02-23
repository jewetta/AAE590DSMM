import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#load data
file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 2\PropertySpace_csv.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, :4].values #features (stiffness tensor values [C11, C12, C22, C66])
y = df.iloc[:, 4].values  #target (volume fraction)

# split the data for test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# applying Box-Cox transformation to target varabile 
y_train_transformed, lambda_boxcox = boxcox(y_train)

print(f"Box-Cox Lambda: {lambda_boxcox:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(y_train_transformed, bins=30, kde=True)
plt.title("Histogram of VF Values")
plt.ylabel("Count")  
plt.xlabel("VF Values")

#training models (using found optimal alpha values to minimize MSE)
lasso = Lasso(alpha=0.0001, random_state=42)
lasso.fit(X_train_scaled, y_train_transformed)

ridge = Ridge(alpha=0.15, random_state=42)
ridge.fit(X_train_scaled, y_train_transformed)

y_predL_transformed = lasso.predict(X_test_scaled)
y_predR_transformed = ridge.predict(X_test_scaled)

y_predL = inv_boxcox(y_predL_transformed, lambda_boxcox)
y_predR = inv_boxcox(y_predR_transformed, lambda_boxcox)

# metrics
mseL = mean_squared_error(y_test, y_predL)
r2L = r2_score(y_test, y_predL)
mseR = mean_squared_error(y_test, y_predR)
r2R = r2_score(y_test, y_predR)

print("\nLasso Model:")
print(f"Mean Squared Error: {mseL:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseL):.4f}")
print(f"R² Score: {r2L:.4f}")

print("\nRidge Model:")
print(f"Mean Squared Error: {mseR:.4f}")
print(f"Root Mean Squared Error Score: {np.sqrt(mseR):.4f}")
print(f"R² Score: {r2R:.4f}")

residualsL = y_test - y_predL
residualsR = y_test - y_predR

#QQ plot of residuals 
plt.figure(figsize=(6, 6))
stats.probplot(residualsL, dist="norm", plot=plt)
plt.xlabel("Theoretical quantiles")  
plt.ylabel("Residuals") 
plt.title("QQ Plot of Residuals for Lasso Regression (After Box-Cox)")
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(residualsR, dist="norm", plot=plt)
plt.xlabel("Theoretical quantiles")  
plt.ylabel("Residuals") 
plt.title("QQ Plot of Residuals for Ridge Regression (After Box-Cox)")
plt.show()
