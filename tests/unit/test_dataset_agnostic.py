from pathlib import Path

FORBIDDEN = ["taxi", "borough", "tpep_", "PULocationID", "DOLocationID"]


def test_src_has_no_dataset_specific_strings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[2] / "src" / "analyst"
    files = list(root.rglob("*.py"))
    assert len(files) > 0
    hits = [
        (p, w)
        for p in files
        for w in FORBIDDEN
        if w.lower() in p.read_text().lower()
    ]
    assert hits == []
