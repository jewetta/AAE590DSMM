import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.callbacks import EarlyStopping

import mat73

SEED = 44
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


#2-point corr fun
def two_point_correlation_fft(image):
    """
    Computes the 2D two-point correlation using FFT
    and returns a 2D array with the same shape as the input image.
    """
    f = np.fft.fftn(image)
    f_conj = np.conjugate(f)
    power_spectrum = f * f_conj

    corr_2D = np.fft.ifftn(power_spectrum).real
    corr_2D = np.fft.fftshift(corr_2D)
    return corr_2D

def main():
    #Tabular data 
    file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/PropertySpace.csv'
    df = pd.read_csv(file_path)

    #load binary image data from matlab file (ShapeSpace.mat)
    mat_data = mat73.loadmat('/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/ShapeSpace.mat')
    ShapeSpace = mat_data['ShapeSpace']  # shape: (50, 50, num_samples)
    num_samples = ShapeSpace.shape[2]

    #    For each sample's 50×50 binary image, compute 2D two-point correlation
    #    and flatten to a 2500-element vector.
    #    corr_vectors will have shape: (num_samples, 2500)
    corr_vectors = []
    for i in range(num_samples):
        image_2d = ShapeSpace[:, :, i]          # (50, 50) for sample i
        corr_2d = two_point_correlation_fft(image_2d) 
        corr_vectors.append(corr_2d.flatten())

    corr_vectors = np.array(corr_vectors)

    #do PCA
    pca = PCA(n_components=15)
    corr_pca = pca.fit_transform(corr_vectors) 

    #Scree Plot of PCA
    plt.figure()
    plt.plot(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        marker='o'
    )
    plt.title('Scree Plot (PCA on Two-Point Correlation)')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    #DATA FROM TABLE (3 stifness tensors and 1 VF)
    X_tabular = df.iloc[:, 2:6].values  
    y = df.iloc[:, 1].values.reshape(-1, 1) #target var

    #Add PCA data (15 more vars)
    X = np.hstack([corr_pca, X_tabular]) #X has 19 independent vars now

    #Train and test data set split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=SEED)

    #Data processing
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    print(f"X_train_scaled shape: {X_train_scaled.shape}")
  
    #define model
    model = keras.Sequential()
    model.add(layers.Dense(
        128,
        activation='relu',
        input_shape=(X_train_scaled.shape[1],),
        kernel_regularizer=keras.regularizers.l1(0.001)
    ))
    model.add(layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=keras.regularizers.l1(0.001)
    ))
    model.add(layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=keras.regularizers.l1(0.001)
    ))
    model.add(layers.Dense(1, activation='linear'))

    #build model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-6),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    #early stop to effectibly choose optimal epoch number
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.01,
        restore_best_weights=True
    )

    #train model
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    #training vs validation loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs. Validation Loss')
    plt.show()

    #evaluate using test set of data 
    test_loss, test_mse = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    print(f"\n[Model] Test MSE (scaled): {test_mse:.4f}")

    #inverse transfrom
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)

    mse_unscaled = mean_squared_error(y_test_orig, y_test_pred)
    r2_test = r2_score(y_test_orig, y_test_pred)
    rmse_unscaled = np.sqrt(mse_unscaled)

    print('-------------------------------------------------------')
    print(f"[Model] Test MSE: {mse_unscaled:.4f}")
    print(f"[Model] Test R^2: {r2_test:.4f}")
    print(f"[Model] Test RMSE: {rmse_unscaled:.4f}")

    #actual vs predicted using test data set
    plt.figure()
    plt.scatter(y_test_orig, y_test_pred, alpha=0.002)
    min_val = min(np.min(y_test_orig), np.min(y_test_pred))
    max_val = max(np.max(y_test_orig), np.max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Actual C11 Stifness Tensor")
    plt.ylabel("Predicted C11 Stifness Tensor")
    plt.show()

    #residuals
    residuals = y_test_orig - y_test_pred
    #resudial histogram
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.show()
    #resudials v pred
    plt.figure()
    plt.scatter(y_test_pred, residuals, alpha=0.002)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.show()


if __name__ == "__main__":
    main()
