import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mat73

#set global seed for keras internal rng and external rng calls
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#define 2-point corrolation function using FFT method
def two_point_correlation_fft(image):
    f = np.fft.fftn(image)
    f_conj = np.conjugate(f)
    corr_2D = np.fft.ifftn(f * f_conj).real
    return np.fft.fftshift(corr_2D)

def main():
    file_path = r'/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/PropertySpace.csv'
    df = pd.read_csv(file_path)
    mat_data = mat73.loadmat('/Users/austinjewett/Desktop/Purdue/AAE590/HW 5/ShapeSpace.mat') #matlab file w/ microstrucutral data
    ShapeSpace = mat_data['ShapeSpace']
    num_samples = ShapeSpace.shape[2]
    corr_vectors = []
    for i in range(num_samples):
        image_2d = ShapeSpace[:, :, i]
        corr_2d = two_point_correlation_fft(image_2d)
        corr_vectors.append(corr_2d.flatten())
    corr_vectors = np.array(corr_vectors)
    #perform PCA
    pca = PCA(n_components=15)
    corr_pca = pca.fit_transform(corr_vectors)
    #plot PC's fropm PCA analysis
    plt.figure()
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
    plt.title('Scree Plot (PCA on Two-Point Correlation)')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()
    #append PC's to indep vars
    X_tabular = df.iloc[:, 2:6].values  
    y = df.iloc[:, 1].values.reshape(-1, 1)
    X = np.hstack([corr_pca, X_tabular])


    print(f"X_train_scaled shape: {X.shape}")
    
    #implemnt 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_r2_list, cv_adj_r2_list, cv_rmse_list, cv_mae_list = [], [], [], []
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_cv)
        y_train_scaled = scaler_y.fit_transform(y_train_cv)
        X_test_scaled = scaler_X.transform(X_test_cv)
        y_test_scaled = scaler_y.transform(y_test_cv)
        
        #same model archetecture and hyperparameters as defined in HW 5
        model = keras.Sequential()
        model.add(layers.Dense(128, activation='relu',
                                 input_shape=(X_train_scaled.shape[1],),
                                 kernel_regularizer=keras.regularizers.l1(0.001)))
        model.add(layers.Dense(64, activation='relu',
                                 kernel_regularizer=keras.regularizers.l1(0.001)))
        model.add(layers.Dense(16, activation='relu',
                                 kernel_regularizer=keras.regularizers.l1(0.001)))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=64, validation_split=0.2,
                            callbacks=[early_stop], verbose=1)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_orig = scaler_y.inverse_transform(y_test_scaled)
        
        mse_val = mean_squared_error(y_test_orig, y_pred)
        r2_val = r2_score(y_test_orig, y_pred)
        rmse_val = np.sqrt(mse_val)
        mae_val = mean_absolute_error(y_test_orig, y_pred)
        n = len(y_test_orig)
        p = X_test_cv.shape[1]
        adj_r2 = 1 - (1 - r2_val) * (n - 1) / (n - p - 1)
        
        cv_r2_list.append(r2_val)
        cv_adj_r2_list.append(adj_r2)
        cv_rmse_list.append(rmse_val)
        cv_mae_list.append(mae_val)
        
        print(f"Fold {fold} - R²: {r2_val:.4f}, Adjusted R²: {adj_r2:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
        fold += 1
        
    #metrics (averaged over 5 folds)
    avg_r2 = np.mean(cv_r2_list)
    avg_adj_r2 = np.mean(cv_adj_r2_list)
    avg_rmse = np.mean(cv_rmse_list)
    avg_mae = np.mean(cv_mae_list)
    
    print("\n5-Fold CV Results:")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Average Adjusted R²: {avg_adj_r2:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    #plots     
    folds = np.arange(1, 6)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(folds, cv_rmse_list, marker='o', label='RMSE')
    plt.plot(folds, cv_mae_list, marker='o', label='MAE')
    plt.xlabel('Fold Number')
    plt.ylabel('Error')
    plt.title('RMSE & MAE per Fold (2pt-PCA ANN)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(folds, cv_r2_list, marker='o', color='green', label='R²')
    plt.plot(folds, cv_adj_r2_list, marker='o', color='blue', label='Adjusted R²')
    plt.xlabel('Fold Number')
    plt.ylabel('Score')
    plt.title('R² & Adjusted R² per Fold (2pt-PCA ANN)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
