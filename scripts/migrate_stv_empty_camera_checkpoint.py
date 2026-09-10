"""Allow the September 2026 empty-camera fix to reuse successful STV rows.

This migration accepts only the exact before/after event and schema hashes. Every
other fingerprinted source file, input file, and saved output must match.
It preserves the original stage state and does not change result tables.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from malca.products.run_metadata import code_fingerprint
from malca.products.stage_state import (
    StageResult, assert_reusable_stage_state, file_signature, fingerprint_digest,
    read_stage_state, write_stage_state,
)


BEFORE_EVENTS_SHA256 = "6359a24cec46b894fd65e82eaab5ce7f623c0ee4b03a708f4bb293cce05d3da1"
AFTER_EVENTS_SHA256 = "82b835e658d8d8878dfbf221c6c2175f4582c549b8f57fe90cdb61d12a823cc9"
BEFORE_CODE = {
    "stv/events.py": BEFORE_EVENTS_SHA256,
    "products/product_schema.py": "8d55e2e3f146a638e931fb4e2fe9cad63f55f2e9955d6d4b1e3a4fd965778776",
}
AFTER_CODE = {
    "stv/events.py": AFTER_EVENTS_SHA256,
    "products/product_schema.py": "7900ff5eaca6ba3f88e877a690c8059e9c33ecb995465d5b5f110c4bd18e0a17",
}


def migrate(run_dir: Path) -> Path:
    state_path = run_dir / "results/_branch_events/lc_events_stochastic_branch_all_STAGE.json"
    state = read_stage_state(state_path)
    if state is None:
        raise ValueError(f"Missing stage state: {state_path}")
    fingerprint = deepcopy(state["fingerprint"])
    if fingerprint["stage"] != "events_stochastic" or fingerprint["stage_version"] != "3":
        raise ValueError("This migration requires the version-3 stochastic event stage")
    stored_code = fingerprint["code"]
    code_root = Path(__file__).resolve().parents[1] / "malca"
    current_code = code_fingerprint(code_root, list(stored_code))
    if {key: current_code.get(key) for key in AFTER_CODE} != AFTER_CODE:
        raise ValueError("Current source is not the exact empty-camera fix supported here")
    stored_version = {key: stored_code.get(key) for key in BEFORE_CODE}
    if stored_version not in (BEFORE_CODE, AFTER_CODE):
        raise ValueError("Stored source is not a supported version for this migration")
    if current_code != {**stored_code, **AFTER_CODE}:
        raise ValueError("Other pipeline source files changed; cached results cannot be migrated")
    assert_reusable_stage_state(state, fingerprint=fingerprint)
    for signature in fingerprint["inputs"]:
        if file_signature(signature["path"], content_hash="sha256" in signature) != signature:
            raise ValueError(f"Input changed: {signature['path']}")
    if stored_version == AFTER_CODE:
        print(f"Checkpoint already migrated: {state_path}")
        return state_path

    backup = state_path.with_suffix(".before_empty_camera_fix.json")
    with backup.open("xb") as handle:
        handle.write(state_path.read_bytes())
    original_digest = fingerprint_digest(fingerprint)
    fingerprint["code"] = current_code
    fingerprint["digest"] = fingerprint_digest(fingerprint)
    result = deepcopy(state["result"])
    result["metadata"]["empty_camera_fix_migration"] = {
        "previous_fingerprint": original_digest,
        "original_state": str(backup),
        "reason": "Only all-cameras-filtered failures now produce explicit rejection rows",
    }
    write_stage_state(state_path, fingerprint=fingerprint, result=StageResult(**result),
                      outputs=[item["path"] for item in state["outputs"]])
    print(f"Preserved {result['succeeded']} successful results; {result['failed']} sources remain to retry.")
    print(f"Original state saved to {backup}")
    return state_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    migrate(parser.parse_args().run_dir)
