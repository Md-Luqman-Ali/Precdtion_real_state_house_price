# Delhi House Price Prediction

An interactive Streamlit demonstration for estimating Delhi property prices using a saved Decision Tree regression model. The application builds its model input using the feature schema stored in `x.csv`.

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Real%20Estate%20Analytics-0A7B83?style=for-the-badge" alt="Real estate analytics">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

## Project overview

The project demonstrates a tabular regression workflow: collect property attributes, construct a feature row, and pass it to a trained Decision Tree model for prediction.

**Live demo:** Not published yet. A public URL will be added after deployment and a successful smoke test.  
**Application entry point:** `EstatePredictor.py`

## Features

- Streamlit interface for property inputs.
- Numeric inputs for area, BHK and parking.
- Categorical inputs for property type, transaction type and location.
- Feature schema read from `x.csv`.
- Input validation before model inference.

## Technology

Python · Pandas · NumPy · scikit-learn · Joblib · Streamlit

## Key files

| File | Purpose |
|---|---|
| `EstatePredictor.py` | Streamlit prediction interface |
| `DecisionTreeReg.pkl` | Saved Decision Tree model |
| `x.csv` | Feature names and ordering used by the app |
| `requirements.txt` | Python dependencies |

## Run locally

The checked-in dependency versions are older; Python 3.10 is a reasonable starting environment. Model compatibility should be checked against the original training environment.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run EstatePredictor.py
```

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Create an app from this repository and select branch `main`.
3. Set the main file to `EstatePredictor.py`.
4. Select a Python version compatible with the pinned packages and saved model.
5. Confirm `DecisionTreeReg.pkl` and `x.csv` are present in the repository root.
6. Deploy and test a prediction using a location from the app's selector.

Add the public app URL here only after it has been opened and smoke-tested successfully.

## Model implementation and limitations

- The app uses the feature names and order from `x.csv` to construct the prediction row.
- A matching feature count does not by itself prove that feature ordering, encoding or preprocessing matches the original training pipeline. Validate these against the original training process before relying on outputs.
- The prediction unit and target transformation have not been independently confirmed; therefore, outputs should not be interpreted as a verified INR, lakh or crore valuation.
- Saved scikit-learn models can be sensitive to library and Python versions.
- This is an educational machine-learning demonstration, not a property valuation service or financial advice.

## Author

**Md Luqman Ali** · [GitHub](https://github.com/Md-Luqman-Ali) · [LinkedIn](https://www.linkedin.com/in/mdluqmanali/)
