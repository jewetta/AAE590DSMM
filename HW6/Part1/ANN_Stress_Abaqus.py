import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

from tensorflow.keras.callbacks import EarlyStopping

SEED = 48 #used for HW analysis (change if u want but results WILL differ)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


#NEW FOR HW 6

print("\n--- Starting 5-Fold Cross Validation for ANN ---\n")

#load data 
cv_file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 3/DataSet_Stress_Import_csv.csv'
df_cv = pd.read_csv(cv_file_path)

#indep vars
X_cv = df_cv.iloc[:, 0:3].values

#dep var (apply log trans to it)
df_cv.iloc[:, 3] = np.log(df_cv.iloc[:, 3])
y_cv = df_cv.iloc[:, 3].values.reshape(-1, 1)

mae_scores_cv = []
r2_scores_cv = []
rmse_scores_cv = []


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)  


#5-fold CV implemtation
fold_num = 1
for train_index, test_index in kf.split(X_cv):
    print(f"Processing Fold {fold_num}...")
    
    #data split (test/train)
    X_train_cv, X_test_cv = X_cv[train_index], X_cv[test_index]
    y_train_cv, y_test_cv = y_cv[train_index], y_cv[test_index]
    
    #std both indep and dep vars for ann build
    scaler_X_cv = StandardScaler()
    scaler_y_cv = StandardScaler()
    
    X_train_cv_scaled = scaler_X_cv.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler_X_cv.transform(X_test_cv)
    y_train_cv_scaled = scaler_y_cv.fit_transform(y_train_cv)
    y_test_cv_scaled = scaler_y_cv.transform(y_test_cv)

    
    #build model (same hyperparams as set in HW 5)
    cv_model = keras.Sequential()
    cv_model.add(layers.Dense(
        4,
        activation='relu',
        input_shape=(X_train_cv_scaled.shape[1],),
        kernel_regularizer=keras.regularizers.l1(0.0001)
    ))
    cv_model.add(layers.Dense(
        2,
        activation='relu',
        kernel_regularizer=keras.regularizers.l1(0.0001)
    ))
    cv_model.add(layers.Dense(
        2,
        activation='relu',
        kernel_regularizer=keras.regularizers.l1(0.0001)
    ))
    cv_model.add(layers.Dense(1, activation='linear'))
    
    cv_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=4e-4),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    
    #early stoping for optimal epoch number
    early_stop_cv = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    #train model
    cv_history = cv_model.fit(
        X_train_cv_scaled,
        y_train_cv_scaled,
        epochs=1000,
        batch_size=5,
        validation_split=0.2,
        callbacks=[early_stop_cv],
        verbose=0
    )
    
    #predictions
    y_test_pred_scaled_cv = cv_model.predict(X_test_cv_scaled)
    
    #inverse transfroms
    y_test_pred_log_cv = scaler_y_cv.inverse_transform(y_test_pred_scaled_cv)
    y_test_orig_log_cv = scaler_y_cv.inverse_transform(y_test_cv_scaled)

    y_test_pred_cv = np.exp(y_test_pred_log_cv)
    y_test_orig_cv = np.exp(y_test_orig_log_cv)
    
    #metrics
    fold_mae = mean_absolute_error(y_test_orig_cv, y_test_pred_cv)
    fold_r2 = r2_score(y_test_orig_cv, y_test_pred_cv)
    fold_rmse = np.sqrt(mean_squared_error(y_test_orig_cv, y_test_pred_cv))
    
    mae_scores_cv.append(fold_mae)
    r2_scores_cv.append(fold_r2)
    rmse_scores_cv.append(fold_rmse)
    
    print(f"Fold {fold_num} - MAE: {fold_mae:.4f}, R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}\n")
    fold_num += 1

#average metrics across 5 folds
avg_mae_cv = np.mean(mae_scores_cv)
avg_r2_cv = np.mean(r2_scores_cv)
avg_rmse_cv = np.mean(rmse_scores_cv)

print("--- Average 5-Fold Cross Validation Metrics (ANN) ---")
print(f"Average MAE: {avg_mae_cv:.4f}")
print(f"Average R²: {avg_r2_cv:.4f}")
print(f"Average RMSE: {avg_rmse_cv:.4f}")

#plots
folds = np.arange(1, 6)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(folds, mae_scores_cv, marker='o', label='MAE')
plt.plot(folds, rmse_scores_cv, marker='o', label='RMSE')
plt.xlabel('Fold Number')
plt.ylabel('Error Metric')
plt.title('MAE & RMSE per Fold (stress) (ANN)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, r2_scores_cv, marker='o', color='green', label='R²')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.title('R² per Fold (stress) (ANN)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
