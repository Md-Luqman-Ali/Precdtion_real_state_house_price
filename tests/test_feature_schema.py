from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[1]

def test_feature_schema_has_unique_columns_and_expected_prefix():
    schema = pd.read_csv(ROOT / "x.csv", nrows=0)
    columns = list(schema.columns)
    assert len(columns) == len(set(columns))
    assert columns[:7] == [
        "Area", "BHK", "Parking", "Apartment",
        "Builder_Floor", "New_Property", "Resale"
    ]
    assert len(columns) > 7
