import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load trained model and metadata
model = joblib.load("outputs/models/demand_forecast_model.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")
features = joblib.load("outputs/models/feature_names.pkl")

# Load the same dataset used in training
listings = pd.read_csv(r"B:\event_aware_forecasting\outputs\event_data\event_aware_listings.csv")

# Ensure event columns exist (safety check)
for col in ['days_to_event', 'is_event_week']:
    if col not in listings.columns:
        listings[col] = 0

# Generate demand score if missing
if 'demand_score' not in listings.columns:
    listings['demand_score'] = (
        listings.get('number_of_reviews', 0) * 0.5
        + (100 - listings.get('availability_30', 100)) * 0.5
    )

# Prepare features and target
X = listings[features].fillna(0)
y = listings['demand_score']
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.4, color='royalblue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Demand (%)")
plt.ylabel("Predicted Demand (%)")
plt.title("📈 Actual vs Predicted Demand (Event-Aware)")
plt.grid(True, linestyle='--', alpha=0.6)

r2 = np.corrcoef(y, y_pred)[0, 1] ** 2
plt.text(y.min(), y.max()*0.9, f"R² ≈ {r2:.2f}", fontsize=12, color='red')
plt.show()
