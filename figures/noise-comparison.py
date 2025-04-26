import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Sample data
ypoints = np.array([15.87, 5.64, 0.60, -4.14, -8.47, -11.87, -13.90, -14.83, -15.17, -15.29])
ypoints1 = np.array([12.13, 6.93, 1.55, -3.87, -8.88, -12.64, -14.71, -15.57, -15.86, -15.96])
ypoints2 = np.array([8.46, 2.90, -3.41, -8.84, -11.32, -12.06, -12.28, -12.36, -12.38, -12.39])
ypoints3 = np.array([-11.55, -11.18, -11.91, -11.27, -9.89, -11.34, -11.60, -11.64, -11.25, -12.11])
ypoints4 = np.array([20, 16, 12.30, 7, 3, -2.5, -7.5, -10.10, -11, -12])
ypoints5 = np.array([4, -5.20, -8, -9.5, -10.2, -11.8, -12.2, -12.8, -12.9, -12.9])
ypoints6 = np.array([np.nan, 0, -5, -10, -14, -16, -16.8, -17.4, -17.5, np.nan])

# List of models including the proposed model
models = [
    ('ACRDNet1x', ypoints),
    ('CRNet', ypoints1),
    ('BCsiNet', ypoints2),
    ('Proposed Model', ypoints3),
    ('CsiNet', ypoints4),
    ('CsiNet+DNNet', ypoints5),
    ('CsiNetPlus', ypoints6),
]

# Calculate Pearson correlation coefficient for the proposed model (ypoints3) with its previous values
valid_indices = ~np.isnan(ypoints3)
valid_model_values = ypoints3[valid_indices]

# Previous values
previous_values = valid_model_values[:-1]
current_values = valid_model_values[1:]

if len(previous_values) > 1:
    corr_proposed_model, _ = pearsonr(previous_values, current_values)
    print(f'Correlation of Proposed Model NMSE with previous value: {corr_proposed_model:.2f}')
else:
    corr_proposed_model = np.nan
    print('Not enough data to calculate correlation for Proposed Model')

# Calculate Pearson correlation coefficients for other models
correlations = []
for model_name, model_values in models:
    if model_name == 'Proposed Model':
        correlations.append((model_name, corr_proposed_model))
        continue
    
    valid_indices = ~np.isnan(model_values)
    valid_model_values = model_values[valid_indices]
    
    # Previous values
    previous_values = valid_model_values[:-1]
    current_values = valid_model_values[1:]
    
    if len(previous_values) > 1:
        corr, _ = pearsonr(previous_values, current_values)
        correlations.append((model_name, corr))
        print(f'Correlation of {model_name} NMSE with previous value: {corr:.2f}')
    else:
        correlations.append((model_name, np.nan))
        print(f'Not enough data to calculate correlation for {model_name}')

# Prepare data for plotting
model_names = [model[0] for model in correlations]
corr_values = [model[1] for model in correlations]

# Plotting the correlation coefficients
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, corr_values, color=['skyblue' if model != 'Proposed Model' else 'orange' for model in model_names])

# Adding value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

# Adding axes labels and title
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
plt.title('Correlation of NMSE Values with Previous Values', fontsize=14, fontweight='bold')

# Adding grid lines for better readability
plt.grid(axis='y', linestyle='--', linewidth=0.7)

# Save the plot as an image file (optional)
plt.savefig('nmse_previous_correlation.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
