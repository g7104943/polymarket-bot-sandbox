import json
import re
from pathlib import Path

import joblib


EXPECTED = {
    "ETH/USDT": {"15m"},
}

SENSITIVE_RE = re.compile(
    r"(PRIVATE_KEY|MNEMONIC|PASSWORD|SECRET|API_KEY|CREDENTIAL|WALLET|/Users/mac|BEGIN [A-Z ]*PRIVATE KEY)",
    re.IGNORECASE,
)


def test_public_model_pack_loads_and_matches_metadata():
    repo_root = Path(__file__).resolve().parents[1]
    models_root = repo_root / "data" / "models"
    seen = {symbol: set() for symbol in EXPECTED}

    for model_dir in sorted(models_root.glob("lightgbm_*_USDT_*_20260121_014635")):
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"
        assert model_path.exists()
        assert meta_path.exists()
        assert model_path.stat().st_size < 10 * 1024 * 1024

        raw_meta = meta_path.read_text(encoding="utf-8")
        assert not SENSITIVE_RE.search(raw_meta)
        metadata = json.loads(raw_meta)
        symbol = metadata["symbol"]
        timeframe = metadata["timeframe"]
        assert symbol in EXPECTED
        assert timeframe in EXPECTED[symbol]
        assert len(metadata.get("feature_names", [])) >= 20

        model = joblib.load(model_path)
        assert hasattr(model, "predict") or hasattr(model, "predict_proba")
        seen[symbol].add(timeframe)

    assert seen == EXPECTED
