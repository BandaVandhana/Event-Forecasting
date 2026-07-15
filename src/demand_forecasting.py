import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_event_data():
    """Load the event-aware listings dataset."""
    listings = pd.read_csv(r"B:\event_aware_forecasting\outputs\event_data\event_aware_listings.csv")
    return listings

def prepare_features(listings):
    """
    Prepare features for demand forecasting.
    Uses review count and availability as a proxy for demand.
    """

    # Helper to safely get numeric column or fill with zeros
    def safe_col(df, col):
        return df[col] if col in df.columns else pd.Series([0]*len(df))

    number_of_reviews = safe_col(listings, 'number_of_reviews').fillna(0)
    availability_30 = safe_col(listings, 'availability_30').fillna(0)

    # Create mock demand metric
    listings['demand_score'] = (number_of_reviews * 0.5 + (100 - availability_30) * 0.5)

    # Choose relevant numeric columns
    features = [
        'price', 'minimum_nights', 'number_of_reviews', 
        'days_to_event', 'is_event_week'
    ]
    features = [col for col in features if col in listings.columns]

    X = listings[features].fillna(0)
    y = listings['demand_score']

    return X, y, features


def train_demand_model(X, y):
    """Train an XGBoost regressor to predict demand."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("✅ Model trained successfully!")
    print(f"📊 MAE: {mae:.2f}")
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 R²: {r2:.2f}")

    return model, scaler


def save_model(model, scaler, features):
    """Save trained model, scaler, and feature names."""
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(model, 'outputs/models/demand_forecast_model.pkl')
    joblib.dump(scaler, 'outputs/models/scaler.pkl')
    joblib.dump(features, 'outputs/models/feature_names.pkl')
    print("✅ Model, scaler, and feature names saved successfully!")


if __name__ == "__main__":
    print("🚀 Starting Demand Forecasting...")
    listings = load_event_data()
    X, y, features = prepare_features(listings)
    model, scaler = train_demand_model(X, y)
    save_model(model, scaler, features)
