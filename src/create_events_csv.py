import pandas as pd
import random
from datetime import datetime, timedelta

# Path to save events.csv
save_path = r"B:\event_aware_forecasting\data\events.csv"

# Some example areas from New York — you can modify them
areas = [
    "Midtown", "Williamsburg", "East Harlem", "Fort Greene",
    "Bedford-Stuyvesant", "Hell's Kitchen", "Long Island City"
]

event_names = [
    "Music Festival", "Tech Conference", "Art Exhibition", "Food Carnival",
    "Marathon", "Film Premiere", "Charity Gala"
]

# Generate random event data
data = []
for _ in range(20):  # 20 random events
    area = random.choice(areas)
    event = random.choice(event_names)
    start_date = datetime(2025, 11, 10) + timedelta(days=random.randint(0, 60))
    end_date = start_date + timedelta(days=random.randint(1, 3))
    latitude = 40.6 + random.random() * 0.2
    longitude = -74.0 + random.random() * 0.1
    influence = round(random.uniform(0.3, 1.0), 2)
    
    data.append([event, start_date.date(), end_date.date(), latitude, longitude, area, influence])

events_df = pd.DataFrame(data, columns=[
    "event_name", "start_date", "end_date", "latitude", "longitude", "area", "influence"
])

events_df.to_csv(save_path, index=False)
print(f"✅ events.csv created at {save_path}")
print(events_df.head())
