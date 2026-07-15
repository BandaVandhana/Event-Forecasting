import pandas as pd
import joblib
from datetime import datetime

from fetch_ticketmaster_events import fetch_ticketmaster_events   # FIXED


# -------------------------------
# 1. Load your trained model
# -------------------------------
model = joblib.load(r"B:\event_aware_forecasting\models\xgb_event_price_model.pkl")
preprocessor = joblib.load(r"B:\event_aware_forecasting\models\preprocessor.pkl")


print("✅ Loaded trained model & preprocessor!")

# -------------------------------
# 2. Fetch LIVE events from Ticketmaster (NYC)
# -------------------------------
events_df = fetch_ticketmaster_events(city="New York", days_ahead=30)

if events_df.empty:
    print("❌ No live events fetched. Exiting.")
    exit()

print("\n🎉 LIVE EVENTS FETCHED FROM TICKETMASTER:")
print(events_df.head())

# -------------------------------
# 3. Preprocess Ticketmaster data
# -------------------------------

def preprocess_ticketmaster(df):
    df = df.copy()

    # Convert date → datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Extract date features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["weekend"] = (df["weekday"] >= 5).astype(int)

    # Standardize columns expected by your model
    df.rename(columns={
        "event_name": "event",
        "genre": "event_category",
    }, inplace=True)

    # Fill missing text fields
    df["event_category"] = df["event_category"].fillna("Unknown")
    df["venue"] = df["venue"].fillna("Unknown")

    # Only keep columns used during training
    final_cols = ["event_category", "venue", "city", "month", "day", "weekday", "weekend"]

    return df[final_cols]

print("\n⚙️ Preprocessing Ticketmaster events...")
X_live = preprocess_ticketmaster(events_df)

# Match training feature format
X_live_transformed = preprocessor.transform(X_live)

# -------------------------------
# 4. Predict ticket prices
# -------------------------------
print("\n📈 Predicting ticket prices for LIVE events...")

events_df["predicted_price"] = model.predict(X_live_transformed)

# -------------------------------
# 5. Save Output
# -------------------------------
events_df.to_csv("data/live_event_predictions.csv", index=False)

print("\n🎉 DONE! Saved: data/live_event_predictions.csv")
print(events_df[["event_name", "date", "genre", "venue", "predicted_price"]].head(10))
