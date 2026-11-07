from pathlib import Path

FORBIDDEN = ["taxi", "borough", "tpep_", "PULocationID", "DOLocationID"]


def test_src_has_no_dataset_specific_strings():
    hits = [
        (p, w)
        for p in Path("src/analyst").rglob("*.py")
        for w in FORBIDDEN
        if w.lower() in p.read_text().lower()
    ]
    assert hits == []
