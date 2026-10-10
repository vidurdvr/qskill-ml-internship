import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Ensure local imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_housing_data
from src.features import HousingFeatureEngineering, add_engineered_features

OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 300

def perform_housing_eda(df: pd.DataFrame):
    """Generate and save comprehensive EDA figures."""
    print("\n--- 1. Exploratory Data Analysis ---")
    
    # 1. Feature Distributions
    features_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedHouseVal']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#3B82F6', bins=40)
        axes[i].set_title(f"Distribution of {col}", fontsize=11, weight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        
    # Remove unused subplots
    for j in range(len(features_to_plot), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("California Housing Feature Distributions", fontsize=15, weight='bold', y=1.01)
    plt.tight_layout()
    dist_path = os.path.join(OUTPUTS_DIR, "feature_distributions.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved: {dist_path}")

    # 2. Correlation Heatmap (including engineered features)
    df_engineered = add_engineered_features(df)
    corr = df_engineered.corr()
    
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of California Housing Attributes", fontsize=14, weight='bold', pad=14)
    plt.tight_layout()
    corr_path = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"Saved: {corr_path}")

    # 3. Median Income vs Price Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.2, c='#2563EB', edgecolors='none', s=15)
    plt.axhline(5.0, color='red', linestyle='--', lw=1.5, label='Price Ceiling ($500,000 Cap)')
    plt.title("Median House Value vs. Median Income", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Median Income ($10,000s)", fontsize=11)
    plt.ylabel("Median House Value ($100,000s)", fontsize=11)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    income_path = os.path.join(OUTPUTS_DIR, "price_vs_income.png")
    plt.savefig(income_path)
    plt.close()
    print(f"Saved: {income_path}")

    # 4. Geographic Map: Lat/Long Scatter Colored by House Price
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        df['Longitude'], df['Latitude'],
        c=df['MedHouseVal'], cmap='plasma',
        s=df['Population'] / 100, alpha=0.5,
        edgecolors='none'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Median House Value ($100,000s)", fontsize=11)
    plt.title("California Housing Geography: Location & Price Gradient\n(Marker size proportional to block population)", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.tight_layout()
    geo_path = os.path.join(OUTPUTS_DIR, "california_geo_scatter.png")
    plt.savefig(geo_path)
    plt.close()
    print(f"Saved: {geo_path}")

def clean_and_prepare_data(df: pd.DataFrame):
    """
    Handle extreme data anomalies and prepare feature matrix X and target y.
    Extreme statistical outliers (e.g. AveRooms > 50 or AveOccup > 50) represent
    data collection artifacts and distort gradient/linear regressions.
    """
    print("\n--- 2. Data Cleaning & Preprocessing ---")
    print(f"Initial row count: {len(df)}")
    print(f"Missing values count:\n{df.isnull().sum()}")
    
    # Cap extreme outliers on non-target features (99.5th percentile or domain bounds)
    data = df.copy()
    
    # Clip extreme averages for stability
    data['AveRooms'] = np.clip(data['AveRooms'], 1.0, 20.0)
    data['AveBedrms'] = np.clip(data['AveBedrms'], 0.5, 10.0)
    data['AveOccup'] = np.clip(data['AveOccup'], 1.0, 20.0)
    data['Population'] = np.clip(data['Population'], 1.0, 15000.0)
    
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    
    return X, y

def train_and_benchmark_models():
    """Run full regression benchmark with 5-fold CV and hyperparameter tuning."""
    raw_df = load_housing_data()
    perform_housing_eda(raw_df)
    
    X, y = clean_and_prepare_data(raw_df)
    
    # 80/20 Train-Test Split (fixed random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Candidate Regressors
    models = {
        'Linear Regression (Baseline)': (
            LinearRegression(),
            {}
        ),
        'Ridge Regression': (
            Ridge(random_state=42),
            {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
        ),
        'Lasso Regression': (
            Lasso(random_state=42, max_iter=3000),
            {'regressor__alpha': [0.001, 0.01, 0.1, 1.0]}
        ),
        'Random Forest Regressor': (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {'regressor__n_estimators': [100, 150], 'regressor__max_depth': [15, 20, None]}
        ),
        'Gradient Boosting Regressor': (
            GradientBoostingRegressor(random_state=42),
            {'regressor__n_estimators': [150, 250], 'regressor__learning_rate': [0.05, 0.1], 'regressor__max_depth': [4, 6]}
        )
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []
    trained_pipelines = {}

    print("\n--- 3. 5-Fold Cross-Validation & Hyperparameter Tuning ---")
    for name, (reg, param_grid) in models.items():
        print(f"Tuning & evaluating {name}...")
        
        # Build pipeline: Feature Engineering -> Scaler -> Regressor
        pipeline = Pipeline([
            ('feat_eng', HousingFeatureEngineering()),
            ('scaler', StandardScaler()),
            ('regressor', reg)
        ])
        
        if param_grid:
            grid = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_
            best_cv_rmse = -grid.best_score_
            best_params = str(grid.best_params_)
        else:
            pipeline.fit(X_train, y_train)
            best_pipe = pipeline
            best_cv_rmse = np.nan
            best_params = "Default"

        y_pred = best_pipe.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        benchmark_results.append({
            'Model': name,
            'CV RMSE ($100k)': best_cv_rmse,
            'Test MSE': mse,
            'Test RMSE ($100k)': rmse,
            'Test RMSE ($)': rmse * 100000,
            'Test MAE ($100k)': mae,
            'Test MAE ($)': mae * 100000,
            'Test R² Score': r2,
            'Best Parameters': best_params
        })
        
        trained_pipelines[name] = {
            'pipeline': best_pipe,
            'y_pred': y_pred,
            'rmse': rmse,
            'r2': r2
        }

    results_df = pd.DataFrame(benchmark_results).sort_values('Test R² Score', ascending=False).reset_index(drop=True)
    summary_path = os.path.join(OUTPUTS_DIR, "metrics_summary.csv")
    results_df.to_csv(summary_path, index=False)
    
    print("\n--- Housing Model Benchmark Summary ---")
    print(results_df[['Model', 'Test RMSE ($100k)', 'Test MAE ($100k)', 'Test R² Score', 'Test RMSE ($)']].to_string(index=False))
    print(f"\nSaved metrics summary: {summary_path}")

    # Best Model Selection
    best_model_name = results_df.iloc[0]['Model']
    best_entry = trained_pipelines[best_model_name]
    best_pipeline = best_entry['pipeline']
    print(f"\n>>> Best Regression Model: {best_model_name} (R² = {best_entry['r2']:.4f}, RMSE = ${best_entry['rmse']*100000:,.2f}) <<<")

    # Save Best Pipeline
    best_model_file = os.path.join(OUTPUTS_DIR, "best_house_price_pipeline.joblib")
    joblib.dump(best_pipeline, best_model_file)
    print(f"Saved best model artifact: {best_model_file}")

    # 4. Evaluation Plots
    print("\n--- 4. Visual Evaluation: Actual vs Predicted & Residuals ---")
    
    # 4A. Actual vs Predicted Scatter
    y_test_arr = y_test.values
    y_pred_arr = best_entry['y_pred']
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_arr, y_pred_arr, alpha=0.25, color='#2563EB', s=20, edgecolors='none')
    plt.plot([0, 5.2], [0, 5.2], color='red', linestyle='--', lw=2, label='Perfect Prediction (y = x)')
    plt.title(f"Actual vs. Predicted House Prices\n{best_model_name}", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Actual Price ($100,000s)", fontsize=11)
    plt.ylabel("Predicted Price ($100,000s)", fontsize=11)
    plt.xlim(0, 5.5)
    plt.ylim(0, 5.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    act_vs_pred_path = os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png")
    plt.savefig(act_vs_pred_path)
    plt.close()
    print(f"Saved: {act_vs_pred_path}")

    # 4B. Residuals Plot & Distribution
    residuals = y_test_arr - y_pred_arr
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred_arr, residuals, alpha=0.25, color='#7C3AED', s=18, edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[0].set_title("Residuals vs. Fitted Values", fontsize=12, weight='bold')
    axes[0].set_xlabel("Predicted Values ($100,000s)")
    axes[0].set_ylabel("Residuals (Actual - Predicted)")
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#7C3AED', bins=40)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_title("Residuals Distribution (Error Spread)", fontsize=12, weight='bold')
    axes[1].set_xlabel("Residual Value ($100,000s)")
    axes[1].set_ylabel("Frequency")
    
    plt.suptitle(f"Residual Diagnostic Analysis: {best_model_name}", fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    resid_path = os.path.join(OUTPUTS_DIR, "residuals_plot.png")
    plt.savefig(resid_path)
    plt.close()
    print(f"Saved: {resid_path}")

    # 4C. Model Benchmark Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    sns.barplot(x='Test R² Score', y='Model', data=results_df, palette='Blues_r', hue='Model', legend=False)
    plt.title("Model Comparison: Test R² Performance", fontsize=13, weight='bold', pad=12)
    plt.xlabel("R² Score (Coefficient of Determination)", fontsize=11)
    plt.ylabel("")
    for i, r2_val in enumerate(results_df['Test R² Score']):
        plt.text(r2_val + 0.01, i, f"{r2_val:.3f}", va='center', fontsize=10, weight='bold')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    bench_chart_path = os.path.join(OUTPUTS_DIR, "model_benchmark_comparison.png")
    plt.savefig(bench_chart_path)
    plt.close()
    print(f"Saved: {bench_chart_path}")

    # 5. Feature Importance Analysis
    print("\n--- 5. Feature Importance Analysis ---")
    regressor = best_pipeline.named_steps['regressor']
    feat_eng = best_pipeline.named_steps['feat_eng']
    feature_names = feat_eng.get_feature_names_out()
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        feat_imp_path = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
        feat_imp_df.to_csv(feat_imp_path, index=False)
        
        plt.figure(figsize=(9, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', hue='Feature', legend=False)
        plt.title(f"Feature Importance ({best_model_name})", fontsize=13, weight='bold', pad=12)
        plt.xlabel("Relative Gini / Split Importance", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        imp_plot_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
        plt.savefig(imp_plot_path)
        plt.close()
        print(f"Saved: {imp_plot_path}")
    
    print("\n==============================================")
    print("California Housing Training & Evaluation Done!")
    print("==============================================")

if __name__ == "__main__":
    train_and_benchmark_models()
