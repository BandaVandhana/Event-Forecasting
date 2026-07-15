import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================
DATA_PATH = r"B:\event_aware_forecasting\data\features.csv"
TARGET_COL = "adjusted_price"

# ===========================
# LOAD DATA
# ===========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

# Features / Target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Identify numeric + categorical
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# ===========================
# PREPROCESSOR (with IMPUTERS)
# ===========================
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

# ===========================
# MODELS TO TEST
# ===========================
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "RandomForest": RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# ===========================
# TRAIN/TEST SPLIT
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# TRAIN & EVAL
# ===========================
results = []

print("\n========== BASELINE MODEL COMPARISON ==========\n")

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{name} → R2: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\n\n========== FINAL RESULTS ==========\n")
print(results_df)

# ===========================
# PLOT RESULTS
# ===========================
plt.figure(figsize=(10, 6))
plt.plot(results_df["Model"], results_df["R2"], marker='o')
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_df["Model"], results_df["RMSE"], marker='o')
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_df["Model"], results_df["MAE"], marker='o')
plt.title("Model Comparison (MAE)")
plt.ylabel("MAE")
plt.grid(True)
plt.show()

