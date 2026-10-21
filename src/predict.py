"""Load the saved model bundle and predict house prices.

The bundle is self-contained: it stores the model, the training columns,
the list of cities and each city's average coordinates, so this module
works in any script or app without needing the training DataFrame.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "house_price_model.joblib"


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def predict_price(bundle, bhk, sqft, city, posted_by="Owner",
                  under_construction=0, rera=0, ready_to_move=1, resale=1,
                  lat=None, lon=None):
    """Return the predicted price in lakhs of rupees."""
    city = city if city in bundle["cities"] else "Other"

    # If no coordinates are given, use the city's average coordinates
    if lat is None or lon is None:
        lat, lon = bundle["city_coords"][city]

    row = pd.DataFrame([{
        "BHK": bhk, "SQUARE_FT": sqft, "UNDER_CONSTRUCTION": under_construction,
        "RERA": rera, "READY_TO_MOVE": ready_to_move, "RESALE": resale,
        "LATITUDE": lat, "LONGITUDE": lon, "POSTED_BY": posted_by, "CITY": city,
    }])
    row = pd.get_dummies(row, columns=["POSTED_BY", "CITY"])
    # Columns the model saw in training but not present here become 0
    # (this is also how the dropped baseline categories are represented)
    row = row.reindex(columns=bundle["columns"], fill_value=0)

    # The model predicts log(1 + price), so convert back to lakhs
    return float(np.expm1(bundle["model"].predict(row))[0])
