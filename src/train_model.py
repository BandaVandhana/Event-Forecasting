import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ======================================================
# 1️⃣ Load Dataset
# ======================================================
print("📂 Loading dataset...")
data_path = r"B:\event_aware_forecasting\data\features.csv"
data = pd.read_csv(data_path)
print(f"✅ Loaded {data.shape[0]} rows and {data.shape[1]} columns")

# ======================================================
# 2️⃣ Data Preparation
# ======================================================
target_col = "adjusted_price"
data = data.dropna(subset=[target_col])

# Drop clear non-predictive or ID columns
drop_cols = ['price', 'log_price', 'id', 'host_id', 'date', 'city', 'area']
for c in drop_cols:
    if c in data.columns:
        data = data.drop(columns=[c])

# Identify object (string) columns
cat_cols = data.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print(f"🧩 Encoding categorical columns: {list(cat_cols)}")
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

# Split features/target
X = data.drop(columns=[target_col])
y = data[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ======================================================
# 3️⃣ Model Training
# ======================================================
print("🚀 Training XGBoost Regressor with regularization...")
xgb_model = XGBRegressor(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# ======================================================
# 4️⃣ Model Evaluation
# ======================================================
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# ======================================================
# 5️⃣ Feature Importance Visualization
# ======================================================
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = feature_importance.head(15)
plt.figure(figsize=(8, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
plt.xlabel("Importance Score")
plt.title("Top 15 Feature Importances (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r"B:\event_aware_forecasting\outputs\feature_importance.png")
plt.close()

print("📊 Feature importance plot saved at: outputs/feature_importance.png")

# ======================================================
# 6️⃣ Save Model
# ======================================================
MODEL_PATH = r"B:\event_aware_forecasting\models\xgb_event_price_model.pkl"
joblib.dump(xgb_model, MODEL_PATH)
print(f"💾 Model saved successfully at: {MODEL_PATH}")

# ======================================================
# 7️⃣ Sample Prediction Check
# ======================================================
sample_pred = xgb_model.predict(X_test.head(5))
print("\n🔍 Sample Predictions:")
for i, val in enumerate(sample_pred):
    print(f"Sample {i+1}: Predicted adjusted_price = {val:.2f}")
