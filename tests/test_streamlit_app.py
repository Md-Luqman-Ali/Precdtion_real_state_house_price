from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_streamlit_app_exposes_prediction_form():
    source = (ROOT / "EstatePredictor.py").read_text(encoding="utf-8")
    assert 'st.form("house_inputs")' in source
    assert 'st.form_submit_button("Estimate price"' in source
    assert 'st.selectbox("Location", locations)' in source
    assert "def build_features(" in source
    assert "def predict_price(" in source


def test_app_uses_schema_order_and_checks_model_shape():
    source = (ROOT / "EstatePredictor.py").read_text(encoding="utf-8")
    assert 'columns = list(schema.columns)' in source
    assert "expected != len(columns)" in source
    assert "columns.index(feature)" in source
