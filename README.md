# Delhi House Price Prediction

Streamlit application for estimating Delhi residential property prices with a saved Decision Tree regression model.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run EstatePredictor.py
```

Key assets: `DecisionTreeReg.pkl`, `x.csv` (ordered feature schema), `Delhi House.csv`, `processed_data.csv`, and `real.jpg`.

The app checks the model's feature count against the CSV schema. Matching counts do not prove feature ordering; confirm the original training pipeline before relying on estimates. Pickled models can require the training scikit-learn version.
