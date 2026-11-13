import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np
data1 = pd.read_csv("Delhi House.csv")
data2= pd.read_csv("processed_data.csv")
x = pd.read_csv("x.csv")
model = joblib.load("DecisionTreeReg.pkl")

if not x.columns.is_unique:
    raise ValueError("Feature schema contains duplicate columns.")

expected_features = getattr(model, "n_features_in_", None)
if expected_features is not None and expected_features != len(x.columns):
    raise ValueError(
        f"Model expects {expected_features} features, but x.csv contains {len(x.columns)}."
    )

model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None and list(model_feature_names) != list(x.columns):
    raise ValueError("Model feature names do not match x.csv column order.")
def predict_price(location, BHK, Parking, Area, Type, Transaction):
    required = [Type, Transaction, location]
    missing = [name for name in required if name not in x.columns]
    if missing:
        raise ValueError(f"Unknown model features: {missing}")

    X = np.zeros((1, len(x.columns)), dtype=float)
    X[0, x.columns.get_loc("Area")] = Area
    X[0, x.columns.get_loc("BHK")] = BHK
    X[0, x.columns.get_loc("Parking")] = Parking

    for name in required:
        X[0, x.columns.get_loc(name)] = 1

    return model.predict(X)[0]

