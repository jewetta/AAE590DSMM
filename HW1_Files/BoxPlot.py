import pandas as pd
import matplotlib.pyplot as plt

file_paths = [
    r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 1\austin_abaqus_data.csv",
    r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 1\dataset_1_abaqus.csv",
    r"C:\Users\33873\Desktop\AAE 590 DSMM\HW 1\dataset_2_abaqus.csv"
]

datasets = [pd.read_csv(file) for file in file_paths]

columns = datasets[0].columns[1:]

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()

for i, col in enumerate(columns):
    data_to_plot = [df[col] for df in datasets]
    
    
    axes[i].boxplot(data_to_plot, tick_labels=['Austin', 'Dataset 1', 'Dataset 2'])
    axes[i].set_title(col)
    axes[i].grid(True)

plt.tight_layout()

save_path = r'C:\Users\33873\Desktop\AAE 590 DSMM\HW 1\boxplot_comparison.png'  
plt.savefig(save_path)

plt.show()