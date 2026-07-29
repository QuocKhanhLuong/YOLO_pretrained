from __future__ import annotations

from pathlib import Path


def test_waruav_suite_defaults_to_data_v3_1_and_batch_8() -> None:
    script = Path("scripts/run_waruav_v3_suite.sh").read_text(encoding="utf-8")

    assert 'DATASET="${DATASET:-data/versions/data_v3.1}"' in script
    assert 'WORKERS="${WORKERS:-4}"' in script
    assert 'BATCH_11S="${BATCH_11S:-8}"' in script
    assert 'BATCH_11M="${BATCH_11M:-8}"' in script
    assert 'BATCH_26S="${BATCH_26S:-8}"' in script
    assert 'BATCH_26M="${BATCH_26M:-8}"' in script
    assert "DATASET_TAG" in script
    assert "data_v3_0" not in script
    assert "data_v3.0" not in script
