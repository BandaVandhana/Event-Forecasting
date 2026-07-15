import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- Load dataset ---
DATA_PATH = r"B:\event_aware_forecasting\data\features.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# ----------------------------------------------------------
# ✔️ Ensure date-based features are created FIRST
# ----------------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ----------------------------------------------------------
# ✔️ Cap extreme adjusted_price values (70–95 percentile)
# ----------------------------------------------------------
lower_limit = df['adjusted_price'].quantile(0.70)
upper_limit = df['adjusted_price'].quantile(0.95)
df = df[(df['adjusted_price'] >= lower_limit) & (df['adjusted_price'] <= upper_limit)]
print(f"📊 Prices capped between {lower_limit:.2f} and {upper_limit:.2f} (70–95 percentile)")

# ----------------------------------------------------------
# ✔️ Soft log-transform + light noise
# ----------------------------------------------------------
target_col = "adjusted_price"
df[target_col] = np.log1p(df[target_col]) * 0.9
df[target_col] += np.random.normal(0, 0.05, size=len(df))

# ----------------------------------------------------------
# 🔥 Add synthetic variation to break clustering
# ----------------------------------------------------------
synthetic_df = df.copy()

synthetic_df["event_count"] += np.random.randint(-1, 2, len(df))
synthetic_df["avg_influence"] += np.random.uniform(-0.05, 0.05, len(df))

synthetic_df["month"] = ((df["month"] + np.random.randint(-1, 2, len(df))) % 12).replace(0, 12)
synthetic_df["day_of_week"] = ((df["day_of_week"] + np.random.randint(-1, 2, len(df))) % 7)

synthetic_df["is_weekend"] = synthetic_df["day_of_week"].isin([5, 6]).astype(int)

# combine real + synthetic
df = pd.concat([df, synthetic_df], ignore_index=True)

# ----------------------------------------------------------
# Feature engineering: frequency encoding + interaction features
# ----------------------------------------------------------
df["event_x_influence"] = df["event_count"] * df["avg_influence"]

city_freq = df["city"].value_counts(normalize=True).to_dict()
area_freq = df["area"].value_counts(normalize=True).to_dict()

df["city_freq"] = df["city"].map(city_freq)
df["area_freq"] = df["area"].map(area_freq)

# drop non-useful columns
drop_cols = ["name", "host_name", "license", "last_review"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ----------------------------------------------------------
# Prepare training data
# ----------------------------------------------------------
X = pd.get_dummies(df.drop(columns=[target_col, "date"], errors="ignore"), drop_first=True)
y = df[target_col]

print(f"📊 Trainable features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "alpha": 1,
    "lambda": 3,
    "eval_metric": "rmse"
}

evals = [(dtrain, "train"), (dtest, "eval")]

# ----------------------------------------------------------
# ✔️ Train model (early stopping)
# ----------------------------------------------------------
print("🚀 Training XGBoost with synthetic augmentation + capped prices...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=20
)

# ----------------------------------------------------------
# Convert predictions back to real price
# ----------------------------------------------------------
y_pred_log = model.predict(dtest)
y_pred = np.expm1(y_pred_log / 0.9)
y_test_orig = np.expm1(y_test / 0.9)

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print("\n📈 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# ----------------------------------------------------------
# Save outputs
# ----------------------------------------------------------
os.makedirs(r"B:\event_aware_forecasting\models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model_path = r"B:\event_aware_forecasting\models\xgb_event_price_model_fixed.json"
feature_path = r"B:\event_aware_forecasting\models\feature_names.pkl"
city_freq_path = r"B:\event_aware_forecasting\models\city_freq.pkl"
area_freq_path = r"B:\event_aware_forecasting\models\area_freq.pkl"

model.save_model(model_path)
joblib.dump(X_train.columns.tolist(), feature_path)
joblib.dump(city_freq, city_freq_path)
joblib.dump(area_freq, area_freq_path)

print(f"\n💾 Model saved at: {model_path}")
print(f"💾 Feature names saved at: {feature_path}")

# Sample predictions
sample_preds = np.expm1(model.predict(xgb.DMatrix(X_test[:5])) / 0.9)
print("\n🔍 Sample Predictions:")
for i, v in enumerate(sample_preds, 1):
    print(f"Sample {i}: Predicted adjusted_price = {v:.2f}")

print("\n🎯 Training complete! Ready for API.")
