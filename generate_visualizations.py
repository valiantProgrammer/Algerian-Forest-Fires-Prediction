"""
Script to generate all visualizations from the notebook and save as PNG files
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create static/visualizations directory if it doesn't exist
import os
os.makedirs('static/visualizations', exist_ok=True)

# Load data
dataset = pd.read_csv('notebooks/Algerian_forest_fires_dataset_UPDATE.csv', header=1)

# Data preprocessing
dataset.loc[:122, "Region"] = 0
dataset.loc[122:, "Region"] = 1
df = dataset
df[['Region']] = df[['Region']].astype(int)
df = df.dropna().reset_index(drop=True)
df = df.drop(122).reset_index(drop=True)
df.columns = df.columns.str.strip()
df[['month','day','year','Temperature','RH','Ws']] = df[['month','day','year','Temperature','RH','Ws']].astype(int)

objects = [features for features in df.columns if df[features].dtypes == 'O']
for i in objects:
    if i != 'Classes':
        df[i] = df[i].astype(float)

df_copy = df.drop(['day','month','year'], axis=1)
df_copy['Classes'] = np.where(df_copy['Classes'].str.contains('not fire'), 0, 1)

# 1. Class Distribution Pie Chart
fig, ax = plt.subplots(figsize=(10, 8))
percentage = df_copy['Classes'].value_counts(normalize=True) * 100
colors = ['#FF6B6B', '#4ECDC4']
explode = (0.05, 0)
ax.pie(percentage, labels=['Fire', 'Not Fire'], autopct='%1.1f%%', 
       colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax.set_title('Dataset Class Distribution', fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/visualizations/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Feature Distribution Histograms
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
features_to_plot = [col for col in df_copy.columns if col != 'Classes'][:9]
for idx, feature in enumerate(features_to_plot):
    row = idx // 3
    col = idx % 3
    axes[row, col].hist(df_copy[feature], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'{feature} Distribution', weight='bold')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/visualizations/02_feature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(14, 10))
correlation_matrix = df_copy.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/visualizations/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Prepare data for modeling
X = df_copy.drop(columns=['FWI'])
y = df_copy['FWI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature correlation and dropping
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

drop_feature = correlation(X_train, 0.85)
X_train = X_train.drop(drop_feature, axis=1)
X_test = X_test.drop(drop_feature, axis=1)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Before and After Standardization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
X_train_df = pd.DataFrame(X_train, columns=X_train.columns)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

sns.boxplot(data=X_train_df, ax=axes[0], color='#3498db')
axes[0].set_title('Before Standardization', fontsize=12, weight='bold')
axes[0].set_ylabel('Value')
axes[0].grid(alpha=0.3)

sns.boxplot(data=X_train_scaled_df, ax=axes[1], color='#2ecc71')
axes[1].set_title('After Standardization', fontsize=12, weight='bold')
axes[1].set_ylabel('Standardized Value')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('static/visualizations/04_standardization_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Train models and store results
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet': ElasticNet(),
    'Ridge CV': RidgeCV(cv=5),
    'Lasso CV': LassoCV(cv=5)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    predictions[name] = y_pred

# 5. Model Comparison - Accuracy Metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['MAE', 'RMSE', 'R2', 'MSE']
model_names = list(results.keys())

for idx, metric in enumerate(metrics):
    row = idx // 2
    col = idx % 2
    values = [results[model][metric] for model in model_names]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(model_names)))
    bars = axes[row, col].bar(range(len(model_names)), values, color=colors, edgecolor='black', linewidth=1.5)
    axes[row, col].set_title(f'{metric} Comparison', fontsize=12, weight='bold')
    axes[row, col].set_xticks(range(len(model_names)))
    axes[row, col].set_xticklabels(model_names, rotation=45, ha='right')
    axes[row, col].set_ylabel(metric)
    axes[row, col].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('static/visualizations/05_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Actual vs Predicted for Ridge Regression (Best Model)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
ridge_pred = predictions['Ridge Regression']
lasso_pred = predictions['Lasso Regression']
linear_pred = predictions['Linear Regression']
elasticnet_pred = predictions['ElasticNet']

for idx, (name, pred) in enumerate([('Linear Regression', linear_pred), 
                                     ('Ridge Regression', ridge_pred),
                                     ('Lasso Regression', lasso_pred),
                                     ('ElasticNet', elasticnet_pred)]):
    row = idx // 2
    col = idx % 2
    axes[row, col].scatter(y_test, pred, alpha=0.6, color='#3498db', edgecolors='black')
    axes[row, col].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
    axes[row, col].set_xlabel('Actual FWI', weight='bold')
    axes[row, col].set_ylabel('Predicted FWI', weight='bold')
    r2 = r2_score(y_test, pred)
    axes[row, col].set_title(f'{name}\nR² = {r2:.4f}', fontsize=11, weight='bold')
    axes[row, col].legend()
    axes[row, col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('static/visualizations/06_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Residuals Plot for Ridge Regression
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for idx, (name, pred) in enumerate([('Linear Regression', linear_pred), 
                                     ('Ridge Regression', ridge_pred),
                                     ('Lasso Regression', lasso_pred),
                                     ('ElasticNet', elasticnet_pred)]):
    row = idx // 2
    col = idx % 2
    residuals = y_test - pred
    axes[row, col].scatter(pred, residuals, alpha=0.6, color='#e74c3c', edgecolors='black')
    axes[row, col].axhline(y=0, color='g', linestyle='--', lw=2)
    axes[row, col].set_xlabel('Predicted FWI', weight='bold')
    axes[row, col].set_ylabel('Residuals', weight='bold')
    axes[row, col].set_title(f'{name} - Residuals Plot', fontsize=11, weight='bold')
    axes[row, col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('static/visualizations/07_residuals_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. MAE and R2 Score Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mae_values = [results[model]['MAE'] for model in model_names]
r2_values = [results[model]['R2'] for model in model_names]

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

ax1.barh(model_names, mae_values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Mean Absolute Error', weight='bold')
ax1.set_title('Model MAE Comparison', fontsize=12, weight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(mae_values):
    ax1.text(v, i, f' {v:.4f}', va='center', fontsize=10)

ax2.barh(model_names, r2_values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('R² Score', weight='bold')
ax2.set_title('Model R² Score Comparison', fontsize=12, weight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(r2_values):
    ax2.text(v, i, f' {v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('static/visualizations/08_mae_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Distribution of Residuals
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for idx, (name, pred) in enumerate([('Linear Regression', linear_pred), 
                                     ('Ridge Regression', ridge_pred),
                                     ('Lasso Regression', lasso_pred),
                                     ('ElasticNet', elasticnet_pred)]):
    row = idx // 2
    col = idx % 2
    residuals = y_test - pred
    axes[row, col].hist(residuals, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[row, col].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[row, col].set_xlabel('Residuals', weight='bold')
    axes[row, col].set_ylabel('Frequency', weight='bold')
    axes[row, col].set_title(f'{name} - Residuals Distribution', fontsize=11, weight='bold')
    axes[row, col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('static/visualizations/09_residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Save model results to a text file for reference
with open('static/visualizations/model_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 80 + "\n\n")
    
    for model_name, metrics in results.items():
        f.write(f"{model_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"  MAE (Mean Absolute Error): {metrics['MAE']:.6f}\n")
        f.write(f"  MSE (Mean Squared Error):  {metrics['MSE']:.6f}\n")
        f.write(f"  RMSE (Root MSE):           {metrics['RMSE']:.6f}\n")
        f.write(f"  R² Score:                  {metrics['R2']:.6f}\n\n")

print("✅ All visualizations generated successfully!")
print(f"📁 Saved to: static/visualizations/")
print("\nGenerated files:")
print("  - 01_class_distribution.png")
print("  - 02_feature_distribution.png")
print("  - 03_correlation_heatmap.png")
print("  - 04_standardization_comparison.png")
print("  - 05_model_comparison.png")
print("  - 06_predictions_comparison.png")
print("  - 07_residuals_plot.png")
print("  - 08_mae_r2_comparison.png")
print("  - 09_residuals_distribution.png")
