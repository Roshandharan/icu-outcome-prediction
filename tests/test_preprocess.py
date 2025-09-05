from src.preprocess import load_data

def test_load_data():
    df = load_data("data/synthetic_icu.csv")
    assert not df.empty