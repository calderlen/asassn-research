from copy import deepcopy
from pathlib import Path

import pytest

from scripts.migrate_stv_empty_camera_checkpoint import (
    AFTER_CODE, BEFORE_CODE, migrate,
)
from malca.products.stage_state import (
    StageResult, assert_reusable_stage_state, build_stage_fingerprint,
    fingerprint_digest, read_stage_state, write_stage_state,
)


@pytest.fixture
def saved_stage(tmp_path):
    source = tmp_path / "source.dat3"
    source.write_text("input photometry")
    output = tmp_path / "events.parquet"
    output.write_bytes(b"existing successful rows")
    code_root = Path(__file__).resolve().parents[1] / "malca"
    fingerprint = build_stage_fingerprint(
        stage="events_stochastic", stage_version="3", candidate_ids=["source"],
        input_paths=[source], settings={"threshold": 5}, code_base=code_root,
        code_paths=["stv/events.py", "products/product_schema.py", "core/utils.py"],
    )
    fingerprint["code"].update(BEFORE_CODE)
    fingerprint["digest"] = fingerprint_digest(fingerprint)
    state_path = tmp_path / "results/_branch_events/lc_events_stochastic_branch_all_STAGE.json"
    write_stage_state(state_path, fingerprint=fingerprint,
                      result=StageResult(stage="events_stochastic", status="partial", expected=1, failed=1),
                      outputs=[output])
    return tmp_path, state_path, source, output


def test_migration_preserves_outputs_and_all_other_fingerprint_fields(saved_stage):
    run_dir, state_path, source, output = saved_stage
    before = state_path.read_bytes()
    original = read_stage_state(state_path)
    expected = deepcopy(original["fingerprint"])
    expected["code"].update(AFTER_CODE)
    expected["digest"] = fingerprint_digest(expected)
    migrate(run_dir)
    migrated = read_stage_state(state_path)
    assert migrated["fingerprint"] == expected
    assert state_path.with_suffix(".before_empty_camera_fix.json").read_bytes() == before
    assert output.read_bytes() == b"existing successful rows"
    assert migrated["result"]["failed"] == 1
    assert_reusable_stage_state(migrated, fingerprint=expected)
    migrated_bytes = state_path.read_bytes()
    migrate(run_dir)
    assert state_path.read_bytes() == migrated_bytes


@pytest.mark.parametrize("changed", ["source", "output", "code", "unsupported_version"])
def test_migration_refuses_unrelated_changes(saved_stage, changed):
    run_dir, state_path, source, output = saved_stage
    state = read_stage_state(state_path)
    if changed == "source":
        source.write_text("changed input")
    elif changed == "output":
        output.write_bytes(b"changed output")
    else:
        key = "core/utils.py" if changed == "code" else "stv/events.py"
        state["fingerprint"]["code"][key] = "unsupported_hash"
        write_stage_state(state_path, fingerprint=state["fingerprint"],
                          result=StageResult(**state["result"]), outputs=[output])
    before = state_path.read_bytes()
    with pytest.raises(ValueError):
        migrate(run_dir)
    assert state_path.read_bytes() == before
    assert not state_path.with_suffix(".before_empty_camera_fix.json").exists()
