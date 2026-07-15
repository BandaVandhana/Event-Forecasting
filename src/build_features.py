import pandas as pd
import numpy as np
from pathlib import Path

# === Paths ===
base = Path(__file__).resolve().parents[1]
merged_path = base / "data" / "merged_airbnb_events.csv"
features_path = base / "data" / "features.csv"

# === Load data ===
df = pd.read_csv(merged_path)
print(f"✅ Loaded merged dataset: {df.shape[0]} rows")

# === Handle missing values ===
df['price'] = df['price'].fillna(df['price'].median())
df['adjusted_price'] = df['adjusted_price'].fillna(df['adjusted_price'].median())

# === Feature engineering ===
df['event_density'] = df['event_count'] / 7.0
df['influence_score'] = df['avg_influence'] * df['total_influence']
df['availability_ratio'] = df['availability_365'] / 365.0
df['log_price'] = np.log1p(df['adjusted_price'])

# === Derived categorical encoding (optional for ML later) ===
df['city_encoded'] = df['city'].astype('category').cat.codes
df['area_encoded'] = df['area'].astype('category').cat.codes

# === Correlation analysis ===
corr = df[['adjusted_price', 'event_count', 'avg_influence',
           'total_influence', 'event_density', 'influence_score',
           'availability_ratio']].corr()

print("\n📊 Correlation with Adjusted Price:")
print(corr['adjusted_price'].sort_values(ascending=False))

# === Save final feature dataset ===
df.to_csv(features_path, index=False)
print(f"\n✅ Feature file saved at {features_path}")
print(f"📈 Total features: {df.shape[1]}")
