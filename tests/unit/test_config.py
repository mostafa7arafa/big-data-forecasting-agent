from analyst.config import Settings


def test_settings_defaults(monkeypatch):
    s = Settings()
    assert s.datasets_dir == s.repo_root / "datasets"
    assert s.artifacts_dir == s.repo_root / "data" / "artifacts"


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", str(tmp_path))
    assert Settings().artifacts_dir == tmp_path
