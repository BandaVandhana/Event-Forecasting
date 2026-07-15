import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("TICKETMASTER_API_KEY")

def fetch_ticketmaster_events(city="New York", days_ahead=30):
    """Fetch upcoming events from Ticketmaster for a specific city."""
    base_url = "https://app.ticketmaster.com/discovery/v2/events.json"
    
    # Define timezone-aware UTC time window
    start_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "apikey": API_KEY,
        "city": city,
        "countryCode": "US",  # US for New York
        "startDateTime": start_date,
        "endDateTime": end_date,
        "size": 50,  # number of events per page
        "sort": "date,asc"
    }

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        print(f"❌ Error fetching events: {response.status_code}")
        return []

    data = response.json()

    # Safely get events
    events = data.get("_embedded", {}).get("events", [])
    if not events:
        print("⚠️ No events found in this period.")
        return []

    result = []
    for event in events:
        # Defensive extraction
        venue_info = event.get("_embedded", {}).get("venues", [{}])[0]
        classification = event.get("classifications", [{}])[0]
        category = classification.get("segment", {}).get("name", "Unknown")

        info = {
            "name": event.get("name", "Unnamed Event"),
            "id": event.get("id", "N/A"),
            "date": event.get("dates", {}).get("start", {}).get("localDate", "Unknown"),
            "venue": venue_info.get("name", "Unknown Venue"),
            "city": venue_info.get("city", {}).get("name", "Unknown City"),
            "category": category
        }
        result.append(info)

    print(f"✅ Retrieved {len(result)} events from Ticketmaster in {city}!")
    return result


if __name__ == "__main__":
    events = fetch_ticketmaster_events(city="New York", days_ahead=15)
    for e in events:
        print(f"{e['date']} | {e['name']} | {e['venue']} | {e['category']}")
