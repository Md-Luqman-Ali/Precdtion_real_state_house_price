# Delhi House Price Prediction

A Streamlit interface for a saved Decision Tree regression model. The app builds its input row in the exact feature order defined by `x.csv`.

## Live demo

Add a verified Streamlit Community Cloud URL here only after the app has been deployed and smoke-tested.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Windows cmd: .venv\\Scripts\\activate.bat
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run EstatePredictor.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io/ and connect GitHub.
3. Create an app from this repository and select `EstatePredictor.py` as the main file.
4. Confirm `DecisionTreeReg.pkl` and `x.csv` remain in the repository root.
5. Deploy and test a prediction using a location from the dropdown.

## Inputs and implementation

- Numeric inputs: area, BHK, and parking.
- Categorical inputs: property type, transaction type, and location.
- The app reads the feature names and ordering from `x.csv`, validates the model's feature count (and feature names when exposed by the estimator), and constructs a single-row NumPy array.
- Existing model and dataset artifacts are preserved.

## Important limitations

- A matching feature count alone does not prove that the feature order or preprocessing matches the original training pipeline. This app follows the checked-in `x.csv` order, but the training pipeline still needs independent verification.
- The output unit is not labelled as INR/crore/lakh because the target transformation and unit have not been confirmed.
- A saved scikit-learn model may require the original Python/scikit-learn versions.
- This is an educational demonstration, not a property valuation or financial advice.

## Stack

Python · Pandas · NumPy · scikit-learn · Joblib · Streamlit
