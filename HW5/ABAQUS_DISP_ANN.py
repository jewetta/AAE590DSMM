import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.callbacks import EarlyStopping

SEED = 48 #used for HW analysis (change if u want but results WILL differ)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/DataSet_Disp_Import_csv.csv' #change with local path for file (data file linked in github repository)
    df = pd.read_csv(file_path)

    X = df.iloc[:, 0:3].values #VF, Load, Mat
    df.iloc[:, 3] = np.log(df.iloc[:, 3])  #Max Disp
    y = df.iloc[:, 3].values.reshape(-1, 1)

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=44)

    #data handeling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    print(f"X_train_scaled shape: {X_train_scaled.shape}")

    #model deff
    model = keras.Sequential()
    model.add(layers.Dense(
        4,
        activation='relu',
        input_shape=(X_train_scaled.shape[1],),
        kernel_regularizer=keras.regularizers.l2(0.0001)
    ))
    model.add(layers.Dense(
        2,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.0001)
    ))
    model.add(layers.Dense(
        2,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.0001)
    ))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=4e-4),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    #early stop to effectibly choose optimal epoch number
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True
    )

    #train model
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=1000,
        batch_size=5,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    #training vs validation loss plot (can tell the early stop is working from this)
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs. Validation Loss')
    plt.show()

    #evaluate model using 'test' data split
    test_loss, test_mse = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    print(f"\n[Model] Test MSE (scaled): {test_mse:.4f}")

    #inverse scaling and transforms
    y_test_pred_scaled = model.predict(X_test_scaled)
    #first inverse_transform to get predictions in log-space
    y_test_pred_log = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_orig_log = scaler_y.inverse_transform(y_test_scaled)
    #then exponentiate to get original stress space
    y_test_pred = np.exp(y_test_pred_log)
    y_test_orig = np.exp(y_test_orig_log)

    mse_unscaled = mean_squared_error(y_test_orig, y_test_pred)
    r2_test = r2_score(y_test_orig, y_test_pred)
    rmse_unscaled = np.sqrt(mse_unscaled)

    print('-------------------------------------------------------')
    print(f"[Model] Test MSE: {mse_unscaled:.4f}")
    print(f"[Model] Test R^2: {r2_test:.4f}")
    print(f"[Model] Test RMSE: {rmse_unscaled:.4f}")


    #test data set: actual vs predicted 
    plt.figure()
    plt.scatter(y_test_orig, y_test_pred, alpha=0.5)
    min_val = min(np.min(y_test_orig), np.min(y_test_pred))
    max_val = max(np.max(y_test_orig), np.max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Actual Max Stress")
    plt.ylabel("Predicted Max Stress")
    plt.show()


if __name__ == "__main__":
    main()
