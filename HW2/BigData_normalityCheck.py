import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

file_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 2\PropertySpace_csv.csv"
df = pd.read_csv(file_path)

independent_vars = df.iloc[:, :4]  
dependent_var = df.iloc[:, 4] 

#plots to check data dist
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(dependent_var, bins=30, kde=True)
plt.title("Histogram of VF Values")
plt.ylabel("Count")  
plt.xlabel("VF Values")

plt.subplot(1, 2, 2)
sns.boxplot(y=dependent_var)
plt.title("Boxplot of VF Values")
plt.ylabel("VF values")  
plt.xlabel(" ") 

plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(dependent_var, dist="norm", plot=plt)
plt.xlabel("Theoretical quantiles")  
plt.ylabel("Ordered values") 
plt.title("Probability Plot")
plt.show()

skewness = skew(dependent_var)
print(f"Skewness: {skewness:.2f}")

