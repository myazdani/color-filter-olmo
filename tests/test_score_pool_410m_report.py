from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from score_pool_410m_report import DEFAULT_C4_PROXY_NOTE, c4_proxy_note  # noqa: E402


def test_c4_proxy_note_falls_back_for_legacy_manifest() -> None:
    manifest = {
        "c4_val_proxy": {
            "label": "c4_val_proxy",
            "source": "allenai/c4 en validation streaming split",
            "path": "/tmp/c4.npy",
        }
    }

    assert c4_proxy_note(manifest) == DEFAULT_C4_PROXY_NOTE


def test_c4_proxy_note_preserves_manifest_note() -> None:
    manifest = {"c4_val_proxy": {"note": "Exact source note."}}

    assert c4_proxy_note(manifest) == "Exact source note."
