import csv
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_feature_schema_has_unique_columns_and_expected_prefix():
    with (ROOT / "x.csv").open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    assert len(columns) == len(set(columns))
    assert columns[:7] == [
        "Area", "BHK", "Parking", "Apartment",
        "Builder_Floor", "New_Property", "Resale"
    ]
    assert len(columns) > 7
