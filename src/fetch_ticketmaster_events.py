# src/fetch_ticketmaster_events.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TICKETMASTER_API_KEY")

def fetch_ticketmaster_events(city=None, country="US", days_ahead=60):
    base_url = "https://app.ticketmaster.com/discovery/v2/events.json"

    start_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "apikey": API_KEY,
        "countryCode": country,
        "startDateTime": start_date,
        "endDateTime": end_date,
        "size": 100,
        "sort": "date,asc"
    }

    if city:
        params["city"] = city

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        print(f"❌ Error fetching events: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    events = data.get("_embedded", {}).get("events", [])

    if not events:
        print(f"⚠️ No events found for {city or country}. Try expanding the date range.")
        return pd.DataFrame()

    rows = []
    for e in events:
        name = e.get("name", "Unknown Event")
        date = e.get("dates", {}).get("start", {}).get("localDate", None)
        genre = (
            e.get("classifications", [{}])[0]
            .get("segment", {})
            .get("name", "Unknown")
        )
        venue = (
            e.get("_embedded", {})
            .get("venues", [{}])[0]
            .get("name", "Unknown Venue")
        )
        city_name = (
            e.get("_embedded", {})
            .get("venues", [{}])[0]
            .get("city", {})
            .get("name", "Unknown")
        )
        rows.append([name, date, genre, venue, city_name])

    df = pd.DataFrame(rows, columns=["event_name", "date", "genre", "venue", "city"])
    df.to_csv("data/raw_ticketmaster_events.csv", index=False)
    print(f"✅ Retrieved {len(df)} events from Ticketmaster ({country}).")
    print(df.head(10))
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df_events = fetch_ticketmaster_events(country="US", days_ahead=90)
