import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color
from sklearn.decomposition import PCA

def two_point_correlation_fft(image):
    """
    Computes the 2D two-point correlation using FFT
    and returns a 2D array with the same shape as the input image.
    Then calculates radial average for comparison plot in part 1 of HW
    """
    #FFT to perform 2-point corrolation
    f = np.fft.fftn(image)
    f_conj = np.conjugate(f)
    power_spectrum = f * f_conj
    corr_2D = np.fft.ifftn(power_spectrum).real
    corr_2D = np.fft.fftshift(corr_2D)

    #radial average of 2-point cor
    ny, nx = corr_2D.shape
    cy, cx = ny // 2, nx // 2
    y_indices, x_indices = np.indices((ny, nx))
    r = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
    r = r.astype(np.int32)
    corr_sum = np.bincount(r.ravel(), weights=corr_2D.ravel())
    r_count = np.bincount(r.ravel())
    radial_profile = corr_sum / r_count
    r_values = np.arange(len(radial_profile))

    return corr_2D, r_values, radial_profile


folder_path = r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 4\Image_Crop" #Change this to local directory file name of these images (https://app.box.com/s/46gg0knlklo1lmb4v0eg2tbww32nyber)

image_files = [f for f in os.listdir(folder_path)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

print("Found image files in folder:")
for img_file in image_files:
    print(img_file)

#need to crop to ensure images are all of same size (I just screen shotted the images so there is a slight discrepancy that must be fixed)
#Crop values found by inspecting images
CROP_HEIGHT = 720
CROP_WIDTH = 3660

all_corr_flat = []
loads = []

fig1, axes1 = plt.subplots(len(image_files), 2, figsize=(12, 3 * len(image_files)))
image_files_sorted = sorted(image_files)
fig2, ax2 = plt.subplots(figsize=(8, 6))

for idx, image_name in enumerate(image_files_sorted):
    image_path = os.path.join(folder_path, image_name)
    image = io.imread(image_path)

    #crop image
    image = image[:CROP_HEIGHT, :CROP_WIDTH]

    load_value = int(image_name.split('N')[0])
    loads.append(load_value)

    #convert image to true grayscale 
    image = color.rgb2gray(image)

    #otsu method to make image binary 
    thresh = filters.threshold_otsu(image)
    binary_image = (image > thresh).astype(np.float32)

    #do 2-point cor
    corr_2D, distances, radial_profile = two_point_correlation_fft(binary_image)

    #flatten for PCA
    all_corr_flat.append(corr_2D.flatten())

    #plots
    axes1[idx, 0].imshow(binary_image, cmap='binary')
    axes1[idx, 0].set_title(f"Binary: {image_name}", fontsize = 8)
    axes1[idx, 0].axis('off')
    im = axes1[idx, 1].imshow(corr_2D, cmap='viridis')
    axes1[idx, 1].set_title("2D Correlation", fontsize = 8)
    axes1[idx, 1].axis('off')
    fig1.colorbar(im, ax=axes1[idx, 1])
    ax2.plot(distances, radial_profile, '-o', label=image_name)
fig1.tight_layout()
ax2.legend()
ax2.set_title("Radial Averaged Two-point Correlation")
ax2.set_xlabel("Distance (pixels)")
ax2.set_ylabel("Correlation")
fig2.tight_layout()
plt.show()

#list to array for PCA
all_corr_flat = np.array(all_corr_flat)  #shape: (9 (num of images), 720*3660)

#do PCA with first 5 components
n_components = 5
pca = PCA(n_components=n_components)
pca_scores = pca.fit_transform(all_corr_flat)  #shape: (9 (num of images), 5)

#show the figure for the first three components 
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    pc_2d = pca.components_[i].reshape((CROP_HEIGHT, CROP_WIDTH))
    im2 = axes2[i].imshow(pc_2d, cmap='viridis')
    axes2[i].set_title(f"PC{i+1}")
    axes2[i].axis('off')
    plt.colorbar(im2, ax=axes2[i], shrink=0.7)
fig2.suptitle("Principal Components of 2-Point Correlation (PC1–PC3)")
plt.tight_layout()
plt.show()

#3d plot with first three PCs
fig3 = plt.figure(figsize=(8,6))
ax3 = fig3.add_subplot(111, projection='3d')
sc = ax3.scatter(pca_scores[:,0], pca_scores[:,1], pca_scores[:,2], c=loads, cmap='plasma', s=60)
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.set_zlabel("PC3")
ax3.set_title("Images in PCA Space (Colored by Load)")
cbar = fig3.colorbar(sc, ax=ax3, label='Load (N)')
plt.tight_layout()
plt.show()

#2d plots comparing PCs w/ each other
fig_2d, axes_2d = plt.subplots(1, 3, figsize=(15, 5))
#1v2
sc_12 = axes_2d[0].scatter(pca_scores[:,0], pca_scores[:,1], c=loads, cmap='plasma', s=60)
axes_2d[0].set_xlabel("PC1")
axes_2d[0].set_ylabel("PC2")
axes_2d[0].set_title("PC1 vs PC2")
fig_2d.colorbar(sc_12, ax=axes_2d[0], label='Load (N)')
#1v3
sc_13 = axes_2d[1].scatter(pca_scores[:,0], pca_scores[:,2], c=loads, cmap='plasma', s=60)
axes_2d[1].set_xlabel("PC1")
axes_2d[1].set_ylabel("PC3")
axes_2d[1].set_title("PC1 vs PC3")
fig_2d.colorbar(sc_13, ax=axes_2d[1], label='Load (N)')
#2v3
sc_23 = axes_2d[2].scatter(pca_scores[:,1], pca_scores[:,2], c=loads, cmap='plasma', s=60)
axes_2d[2].set_xlabel("PC2")
axes_2d[2].set_ylabel("PC3")
axes_2d[2].set_title("PC2 vs PC3")
fig_2d.colorbar(sc_23, ax=axes_2d[2], label='Load (N)')

plt.tight_layout()
plt.show()

#Scree plot w/ first 5 PCA components
explained_variance = pca.explained_variance_ratio_  
x_vals = np.arange(1, n_components+1)

fig4, ax4 = plt.subplots(figsize=(6,4))
ax4.bar(x_vals, explained_variance*100, tick_label=[f'PC{i}' for i in x_vals])
ax4.set_xlabel("Principal Component")
ax4.set_ylabel("Explained Variance (%)")
ax4.set_title("Scree Plot (First 5 PCs)")
plt.tight_layout()
plt.show()

print("Explained variance ratio (first 5 PCs):", explained_variance)

#individual PC plots against load (image)
fig_pc_vs_load, axes_pc_load = plt.subplots(1, 3, figsize=(15, 5))
#1vLoad
axes_pc_load[0].plot(loads, pca_scores[:, 0], marker='o')
axes_pc_load[0].set_title("PC1 vs Load")
axes_pc_load[0].set_xlabel("Load (N)")
axes_pc_load[0].set_ylabel("PC1 Score")
#2vLoad
axes_pc_load[1].plot(loads, pca_scores[:, 1], marker='o')
axes_pc_load[1].set_title("PC2 vs Load")
axes_pc_load[1].set_xlabel("Load (N)")
axes_pc_load[1].set_ylabel("PC2 Score")
#3vLoad
axes_pc_load[2].plot(loads, pca_scores[:, 2], marker='o')
axes_pc_load[2].set_title("PC3 vs Load")
axes_pc_load[2].set_xlabel("Load (N)")
axes_pc_load[2].set_ylabel("PC3 Score")

plt.tight_layout()
plt.show()