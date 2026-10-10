import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features import HousingFeatureEngineering

MODEL_PATH = os.path.join(CURRENT_DIR, "outputs", "best_house_price_pipeline.joblib")

def predict_house_value(
    med_inc: float,
    house_age: float,
    ave_rooms: float,
    ave_bedrms: float,
    population: float,
    ave_occup: float,
    latitude: float,
    longitude: float,
    model_path: str = MODEL_PATH
) -> dict:
    """
    Predict California median house value given census block attributes.
    Target value in California Housing is in $100,000s.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please run 'python src/train.py' first to train and export the pipeline."
        )

    pipeline = joblib.load(model_path)
    
    input_df = pd.DataFrame([{
        'MedInc': float(med_inc),
        'HouseAge': float(house_age),
        'AveRooms': float(ave_rooms),
        'AveBedrms': float(ave_bedrms),
        'Population': float(population),
        'AveOccup': float(ave_occup),
        'Latitude': float(latitude),
        'Longitude': float(longitude)
    }])
    
    predicted_units = float(pipeline.predict(input_df)[0])
    # Enforce realistic lower bound
    predicted_units = max(predicted_units, 0.15)
    predicted_dollars = predicted_units * 100000.0

    return {
        "predicted_value_units": predicted_units,
        "predicted_price_usd": predicted_dollars,
        "input_features": input_df.to_dict(orient='records')[0]
    }

def main():
    parser = argparse.ArgumentParser(
        description="California House Price Prediction CLI"
    )
    parser.add_argument("--MedInc", type=float, default=None, help="Median Income in block group ($10k, e.g. 3.87)")
    parser.add_argument("--HouseAge", type=float, default=None, help="Median house age in years (e.g. 28.0)")
    parser.add_argument("--AveRooms", type=float, default=None, help="Average number of rooms per household (e.g. 5.4)")
    parser.add_argument("--AveBedrms", type=float, default=None, help="Average number of bedrooms per household (e.g. 1.05)")
    parser.add_argument("--Population", type=float, default=None, help="Block group population (e.g. 1425)")
    parser.add_argument("--AveOccup", type=float, default=None, help="Average household occupancy (e.g. 3.0)")
    parser.add_argument("--Latitude", type=float, default=None, help="Latitude (e.g. 37.88)")
    parser.add_argument("--Longitude", type=float, default=None, help="Longitude (e.g. -122.23)")

    args = parser.parse_args()

    # If any required arg is None, use realistic sample defaults or interactive input
    if any(v is None for v in [args.MedInc, args.HouseAge, args.AveRooms, args.AveBedrms, args.Population, args.AveOccup, args.Latitude, args.Longitude]):
        print("=" * 65)
        print("     California House Price Prediction (CLI) — Interactive")
        print("=" * 65)
        print("Note: Values default to typical California census block medians if left blank.")
        
        def prompt_val(prompt, default):
            user_input = input(f"{prompt} [default {default}]: ").strip()
            return float(user_input) if user_input else default

        med_inc = prompt_val("Median Income ($10,000s)", 3.87)
        house_age = prompt_val("Median House Age (years)", 28.0)
        ave_rooms = prompt_val("Average Rooms per household", 5.4)
        ave_bedrms = prompt_val("Average Bedrooms per household", 1.05)
        population = prompt_val("Population", 1425.0)
        ave_occup = prompt_val("Average Occupants per household", 3.0)
        latitude = prompt_val("Latitude (e.g., Bay Area 37.88)", 37.88)
        longitude = prompt_val("Longitude (e.g., Bay Area -122.23)", -122.23)
    else:
        med_inc = args.MedInc
        house_age = args.HouseAge
        ave_rooms = args.AveRooms
        ave_bedrms = args.AveBedrms
        population = args.Population
        ave_occup = args.AveOccup
        latitude = args.Latitude
        longitude = args.Longitude

    res = predict_house_value(
        med_inc=med_inc,
        house_age=house_age,
        ave_rooms=ave_rooms,
        ave_bedrms=ave_bedrms,
        population=population,
        ave_occup=ave_occup,
        latitude=latitude,
        longitude=longitude
    )

    print("\n" + "=" * 55)
    print("           PREDICTION RESULT")
    print("=" * 55)
    print(f"Predicted Median Value:   ${res['predicted_price_usd']:,.2f}")
    print(f"Index Value ($100k):      {res['predicted_value_units']:.4f}")
    print("\nInput Attributes:")
    for k, v in res['input_features'].items():
        print(f"  • {k:<12}: {v}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
