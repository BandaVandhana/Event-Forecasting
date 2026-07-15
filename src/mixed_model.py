import pandas as pd
from datetime import datetime

# === File paths ===
listings_path = r"B:\event_aware_forecasting\data\listings.csv"
events_path = r"B:\event_aware_forecasting\data\events.csv"

# === Load data ===
listings = pd.read_csv(listings_path)
events = pd.read_csv(events_path)

# === Clean listings ===
listings['price'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
listings['city'] = 'New York'
listings['date'] = datetime.now().date()

# Rename neighbourhood column for merging
listings = listings.rename(columns={'neighbourhood': 'area'})

# === Aggregate events by area ===
event_summary = (
    events.groupby('area', as_index=False)
    .agg(
        event_count=('event_name', 'count'),
        avg_influence=('influence', 'mean')
    )
)
event_summary['total_influence'] = event_summary['event_count'] * event_summary['avg_influence']

# === Merge ===
merged = listings.merge(event_summary, on='area', how='left')

# Fill missing influence info for areas without events
merged[['event_count', 'avg_influence', 'total_influence']] = merged[
    ['event_count', 'avg_influence', 'total_influence']
].fillna(0)

# === Optional: Adjust prices based on event influence ===
# (This adds up to +15% variation depending on event impact)
merged['adjusted_price'] = merged['price'] * (1 + 0.15 * merged['avg_influence'])

# === Save merged data ===
output_path = r"B:\event_aware_forecasting\data\merged_airbnb_events.csv"
merged.to_csv(output_path, index=False)

# === Display summary ===
print("📊 Sample merged data (cleaned):")
print(merged[['city', 'area', 'price', 'adjusted_price', 'availability_365',
              'event_count', 'avg_influence', 'total_influence']].head(10))

print(f"\n✅ Merged file saved at {output_path}")
