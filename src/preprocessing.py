import pandas as pd

def clean_listings(file_path):
    listings = pd.read_csv(r"B:\event_aware_forecasting\data\listings.csv")
    
    # Clean price column
    if 'price' in listings.columns:
        listings['price'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
    
    # Convert relevant date columns if they exist
    date_cols = [col for col in listings.columns if 'date' in col.lower()]
    for col in date_cols:
        listings[col] = pd.to_datetime(listings[col], errors='coerce')
    
    # Drop unnecessary columns (optional)
    listings = listings.dropna(subset=['price'])
    
    return listings


def clean_reviews(file_path):
    reviews = pd.read_csv(file_path)
    if 'date' in reviews.columns:
        reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
    return reviews


def save_cleaned_data(listings, reviews):
    listings.to_csv('outputs/cleaned_data/cleaned_listings.csv', index=False)
    reviews.to_csv('outputs/cleaned_data/cleaned_reviews.csv', index=False)
    print("✅ Cleaned data saved successfully!")


