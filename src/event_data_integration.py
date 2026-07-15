import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_cleaned_data():
    listings = pd.read_csv(r"B:\event_aware_forecasting\outputs\cleaned_data\cleaned_listings.csv")
    reviews = pd.read_csv(r"B:\event_aware_forecasting\outputs\cleaned_data\cleaned_reviews.csv")
    return listings, reviews


def generate_mock_event_data(start_date='2024-01-01', end_date='2024-12-31'):
    """
    Mock event data (you’ll replace with API results later).
    Each event has: event_name, event_type, start_date, and city.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='7D')
    event_types = ['Concert', 'Sports', 'Festival', 'Conference', 'Holiday']

    data = {
        'event_name': [f"Event_{i}" for i in range(len(dates))],
        'event_type': np.random.choice(event_types, len(dates)),
        'start_date': dates,
        'city': ['New York'] * len(dates)
    }
    events = pd.DataFrame(data)
    return events


def integrate_event_features(listings, reviews, events):
    """
    Merge event influence — add columns like `days_to_event` or `is_event_week`.
    """
    # Handle missing review dates gracefully
    if 'last_review' in listings.columns:
        listings['date'] = pd.to_datetime(listings['last_review'], errors='coerce')
    else:
        listings['date'] = pd.NaT

    # Replace missing dates with median or a fixed recent date
    median_date = pd.Timestamp('2024-06-01')
    listings['date'] = listings['date'].fillna(median_date)

    events['start_date'] = pd.to_datetime(events['start_date'])

    # Find nearest event safely
    def nearest_event_date(current_date):
        valid_events = events[events['start_date'].notna()]
        if valid_events.empty:
            return np.nan
        diffs = abs(valid_events['start_date'] - current_date)
        nearest_idx = diffs.idxmin()
        return valid_events.loc[nearest_idx, 'start_date']

    listings['next_event_date'] = listings['date'].apply(nearest_event_date)
    listings['days_to_event'] = (listings['next_event_date'] - listings['date']).dt.days
    listings['is_event_week'] = listings['days_to_event'].apply(lambda x: 1 if 0 <= x <= 7 else 0)

    print("✅ Event features integrated successfully!")
    return listings


def save_event_enhanced_data(listings, events):
    os.makedirs('outputs/event_data', exist_ok=True)
    listings.to_csv('outputs/event_data/event_aware_listings.csv', index=False)
    events.to_csv('outputs/event_data/mock_events.csv', index=False)
    print("✅ Event-enhanced data saved successfully!")


if __name__ == "__main__":
    print("🎉 Integrating Event Data...")
    listings, reviews = load_cleaned_data()
    events = generate_mock_event_data()
    enhanced_listings = integrate_event_features(listings, reviews, events)
    save_event_enhanced_data(enhanced_listings, events)
