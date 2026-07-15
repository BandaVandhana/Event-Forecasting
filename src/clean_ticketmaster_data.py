# src/clean_ticketmaster_data.py
import pandas as pd
from datetime import datetime, timedelta

def clean_ticketmaster_data(input_path="data/raw_ticketmaster_events.csv", output_path="data/clean_ticketmaster_events.csv"):
    df = pd.read_csv(input_path)

    # Remove test or empty event names
    df = df.dropna(subset=["event_name"])
    df = df[~df["event_name"].str.contains("test", case=False, na=False)]

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print("🕒 Sample parsed dates:", df["date"].head(5).tolist())

    # Filter out clearly invalid dates
    df = df.dropna(subset=["date"])

    # Allow events from today onwards (±1 day tolerance)
    today = pd.Timestamp.today().normalize()
    df = df[df["date"] >= today - timedelta(days=1)]

    # Add helper columns
    df["days_until_event"] = (df["date"] - today).dt.days.clip(lower=0)
    df["genre"] = df["genre"].fillna("Unknown")
    df["city"] = df["city"].fillna("Unknown")

    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")
    print(f"📊 Total upcoming events: {len(df)}")
    print(df.head(10))
    return df

if __name__ == "__main__":
    clean_ticketmaster_data()
