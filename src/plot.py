import matplotlib.pyplot as plt
import pandas as pd

# ========= YOUR FINAL RESULTS ==========
results = {
    "Model": ["LinearRegression", "Ridge", "Lasso", "RandomForest", "XGBoost_baseline", "Optimized_XGBoost"],
    "R2":    [0.996356, 0.998403, 0.998152, 0.999853, 0.991011, 0.981],
    "MAE":   [99.25, 32.66, 15.71, 0.87, 17.21, 8.95],
    "RMSE":  [203.78, 134.90, 145.11, 40.91, 320.05, 12.16]
}

results_df = pd.DataFrame(results)

# Ensure consistent ordering
results_df = results_df.sort_values(by="Model")

# ===========================
# PLOT R2
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(results_df["Model"], results_df["R2"], marker='o', linewidth=2)
plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ===========================
# PLOT RMSE
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(results_df["Model"], results_df["RMSE"], marker='o', linewidth=2)
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ===========================
# PLOT MAE
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(results_df["Model"], results_df["MAE"], marker='o', linewidth=2)
plt.title("Model Comparison (MAE)")
plt.ylabel("MAE")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
