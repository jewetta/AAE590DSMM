import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.callbacks import EarlyStopping

import os
import shutil
import h5py
import mat73

SEED = 44
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/PropertySpace.csv'
df = pd.read_csv(file_path)

X = df.iloc[:, 2:6].values  
y = df.iloc[:, 1].values.reshape(-1, 1)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

kf = KFold(n_splits=5, shuffle=True, random_state=44)

cv_r2_list = []
cv_adj_r2_list = []
cv_rmse_list = []
cv_mae_list = []

#implemt 5-fold CV
folds = []
fold_num = 1
for train_idx, test_idx in kf.split(X):
    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
    y_train_cv, y_test_cv = y[train_idx], y[test_idx]
    
    scaler_X_cv = StandardScaler()
    scaler_y_cv = StandardScaler()
    X_train_cv_scaled = scaler_X_cv.fit_transform(X_train_cv)
    y_train_cv_scaled = scaler_y_cv.fit_transform(y_train_cv)
    X_test_cv_scaled = scaler_X_cv.transform(X_test_cv)
    y_test_cv_scaled = scaler_y_cv.transform(y_test_cv)
    
    #same ann archetecture and hyperparameters as defined in HW 5
    model_cv = keras.Sequential()
    model_cv.add(layers.Dense(128, activation='relu',
                               input_shape=(X_train_cv_scaled.shape[1],),
                               kernel_regularizer=keras.regularizers.l2(0.001)))
    model_cv.add(layers.Dense(64, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)))
    model_cv.add(layers.Dense(16, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)))
    model_cv.add(layers.Dense(1, activation='linear'))
    
    model_cv.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6),
                     loss='mean_squared_error',
                     metrics=['mean_squared_error'])
    
    early_stop_cv = EarlyStopping(monitor='val_loss', patience=5,
                                  min_delta=0.01, restore_best_weights=True)
    
    model_cv.fit(X_train_cv_scaled, y_train_cv_scaled,
                 epochs=100, batch_size=64, validation_split=0.2,
                 callbacks=[early_stop_cv], verbose=1)
    
    y_pred_cv_scaled = model_cv.predict(X_test_cv_scaled)
    y_pred_cv = scaler_y_cv.inverse_transform(y_pred_cv_scaled)
    y_test_cv_orig = scaler_y_cv.inverse_transform(y_test_cv_scaled)
    
    mse_cv = mean_squared_error(y_test_cv_orig, y_pred_cv)
    r2_cv = r2_score(y_test_cv_orig, y_pred_cv)
    rmse_cv = np.sqrt(mse_cv)
    mae_cv = mean_absolute_error(y_test_cv_orig, y_pred_cv)
    
    n = len(y_test_cv_orig)
    p = X_test_cv.shape[1]
    adj_r2_cv = 1 - (1 - r2_cv) * (n - 1) / (n - p - 1)
    
    cv_r2_list.append(r2_cv)
    cv_adj_r2_list.append(adj_r2_cv)
    cv_rmse_list.append(rmse_cv)
    cv_mae_list.append(mae_cv)
    folds.append(fold_num)
    
    print(f"Fold {fold_num} - R²: {r2_cv:.4f}, Adjusted R²: {adj_r2_cv:.4f}, RMSE: {rmse_cv:.4f}, MAE: {mae_cv:.4f}")
    fold_num += 1

#average metrics over 5 folds
avg_r2 = np.mean(cv_r2_list)
avg_adj_r2 = np.mean(cv_adj_r2_list)
avg_rmse = np.mean(cv_rmse_list)
avg_mae = np.mean(cv_mae_list)

print("\n5-Fold Cross-Validation Results:")
print(f"Average R²: {avg_r2:.4f}")
print(f"Average Adjusted R²: {avg_adj_r2:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")

#plots
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(folds, cv_rmse_list, marker='o', label='RMSE')
plt.plot(folds, cv_mae_list, marker='o', label='MAE')
plt.xlabel('Fold Number')
plt.ylabel('Error')
plt.title('RMSE & MAE per Fold (simple ANN)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(folds, cv_r2_list, marker='o', color='green', label='R²')
plt.plot(folds, cv_adj_r2_list, marker='o', color='blue', label='Adjusted R²')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('R² & Adjusted R² per Fold (simple ANN)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
