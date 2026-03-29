import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_utils import PROCESSED_DATA_PATH
from src.data_utils import load_data
from src.eda_utils import show_summary

def get_insights(df):
    if 'price' in df.columns:
        print("Price Range: ", df['price'].max() ,"-", df['price'].min())
    
    print("✈️ Airline Analysis:\n")
    if 'airline' not in df.columns or 'price' not in df.columns:
        print("Required columns 'Airline' or 'Price' not found.")
        return
    avg_prices = df.groupby("airline")["price"].mean().sort_values(ascending=False)
    print("🔹 Airlines sorted by average ticket price (High ➝ Low):\n")
    print(avg_prices)
    print("\n🔹 Most expensive and cheapest flight from each airline:\n")
    for airline in df['airline'].unique():
        airline_df = df[df['airline'] == airline]
        max_price_row = airline_df.loc[airline_df['price'].idxmax()]
        min_price_row = airline_df.loc[airline_df['price'].idxmin()]
        print(f"\n✈️ Airline: {airline}\n")
        print("💰 Most Expensive Flight: ", max_price_row.to_dict()["flight"])
        print("🪙 Cheapest Flight: ", min_price_row.to_dict()['flight'])
        luxury_flight = airline_df[airline_df['class'] == "Business"]['flight'].unique()
        if len(luxury_flight) > 0:
            print("Luxury flight:\n", luxury_flight)
        else:
            print("No luxury flights available.")

    print("\nRelation of number of stops with price: ", df.groupby("stops")["price"].mean().sort_values(ascending=False))    
    cheapest_flights = df.loc[df.groupby("source_city")["price"].idxmin()]
    print()
    for _, row in cheapest_flights.iterrows():
        print(f"Source City: {row['source_city']}, Cheapest Flight: {row['flight']}, Price: {row['price']}")
    print('\nMost expensive location to travel to: ', df.loc[df['price'].idxmax()]['destination_city'])
    print('\nprice of flights w.r.t. time of a day: ', df.groupby("departure_time")["price"].mean().sort_values(ascending=False))

if __name__ == "__main__":
    df = load_data(PROCESSED_DATA_PATH)
    print("Data loaded successfully.")
    show_summary(df)
    get_insights(df)
    