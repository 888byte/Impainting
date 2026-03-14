from __future__ import annotations

import json

from utils.config_utils import load_config


def test_load_config_accepts_utf8_bom(scratch_dir):
    cfg_path = scratch_dir / "bom_config.json"
    payload = {"train": {"device": "cpu"}}
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8-sig")

    cfg = load_config(str(cfg_path))

    assert cfg["train"]["device"] == "cpu"
