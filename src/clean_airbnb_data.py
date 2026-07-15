import pandas as pd
from datetime import datetime

# Load the Airbnb data
file_path = "data/listings.csv"
df = pd.read_csv(file_path)

# Keep relevant columns
df = df[['neighbourhood_group', 'neighbourhood', 'price', 'availability_365']].copy()

# Rename for consistency
df.rename(columns={
    'neighbourhood_group': 'city',
    'neighbourhood': 'area'
}, inplace=True)

# Drop missing or invalid price/availability values
df.dropna(subset=['price', 'availability_365'], inplace=True)
df = df[df['price'] > 0]

# Add a "date" column (assuming data corresponds to current availability year)
today = datetime.today().date()
df['date'] = pd.to_datetime(today)

# Clean city names to match Ticketmaster event data
df['city'] = df['city'].str.strip().str.title()

# Save cleaned data
output_path = "data/clean_airbnb_listings.csv"
df.to_csv(output_path, index=False)

print(f"✅ Cleaned Airbnb data saved to {output_path}")
print("📊 Sample output:")
print(df.head())
