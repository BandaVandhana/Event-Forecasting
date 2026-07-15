import pandas as pd
from pathlib import Path

# Load cleaned Ticketmaster events
data_path = Path(r"B:\event_aware_forecasting\data\clean_ticketmaster_events.csv")
df = pd.read_csv(data_path)

if df.empty:
    print("⚠️ No event data found! Run fetch and clean scripts first.")
    exit()

# Define base weights by genre
weights = {
    "Music": 1.0,
    "Sports": 0.9,
    "Arts & Theatre": 0.8,
    "Miscellaneous": 0.5,
    "Undefined": 0.3,
    "Unknown": 0.3
}

# Compute influence score
df["genre_weight"] = df["genre"].map(weights).fillna(0.4)
df["event_influence"] = df["genre_weight"] / (1 + df["days_until_event"].astype(float))

# Group by date and city
influence_summary = (
    df.groupby(["city", "date"])
      .agg(event_count=("event_name", "count"),
           avg_influence=("event_influence", "mean"),
           total_influence=("event_influence", "sum"))
      .reset_index()
)

# Save to file
output_path = Path("data/event_influence_summary.csv")
influence_summary.to_csv(output_path, index=False)

print(f"✅ Event influence summary saved to {output_path}")
print("📊 Sample output:")
print(influence_summary.head(10))
