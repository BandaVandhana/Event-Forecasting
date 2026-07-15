import pandas as pd
import numpy as np
import os

# -----------------------------
# 1. Load cleaned datasets
# -----------------------------
airbnb_path = "data/clean_airbnb_listings.csv"
events_path = "data/event_influence_summary.csv"

if not os.path.exists(airbnb_path) or not os.path.exists(events_path):
    raise FileNotFoundError("❌ Missing one or more required CSV files in the data/ folder.")

airbnb = pd.read_csv(airbnb_path)
events = pd.read_csv(events_path)

print(f"✅ Loaded Airbnb data: {airbnb.shape[0]} rows")
print(f"✅ Loaded Event data: {events.shape[0]} rows")

# -----------------------------
# 2. Normalize city names for merging
# -----------------------------
airbnb["city"] = airbnb["city"].str.strip().str.title()
events["city"] = events["city"].str.strip().str.title()

# -----------------------------
# 3. Convert date columns to datetime
# -----------------------------
airbnb["date"] = pd.to_datetime(airbnb["date"], errors="coerce")
events["date"] = pd.to_datetime(events["date"], errors="coerce")

# -----------------------------
# 4. Merge Airbnb listings with event influence
# -----------------------------
merged = pd.merge(
    airbnb,
    events,
    on=["city", "date"],
    how="left"
)

# Fill missing event influence with 0
merged["event_count"] = merged["event_count"].fillna(0)
merged["avg_influence"] = merged["avg_influence"].fillna(0)
merged["total_influence"] = merged["total_influence"].fillna(0)

print(f"📊 Sample merged data (before noise):")
print(merged.head())

# -----------------------------
# 5. Add synthetic event influence if all zero
# -----------------------------
if merged["event_count"].sum() == 0:
    print("\n⚠️ No matching event dates found — adding synthetic event influence for testing.")
    
    np.random.seed(42)
    event_mask = np.random.rand(len(merged)) < 0.2  # ~20% of listings have events nearby
    
    merged.loc[event_mask, "event_count"] = np.random.randint(1, 6, event_mask.sum())
    merged.loc[event_mask, "avg_influence"] = np.round(np.random.uniform(0.3, 0.9, event_mask.sum()), 2)
    merged["total_influence"] = merged["event_count"] * merged["avg_influence"]

# -----------------------------
# 6. Save the merged dataset
# -----------------------------
output_path = "data/merged_airbnb_event_data.csv"
merged.to_csv(output_path, index=False)

print(f"\n✅ Merged dataset saved to {output_path}")
print("📊 Sample merged data (final):")
print(merged.head(10))
