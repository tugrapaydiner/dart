"""Load and reduce bounded synthetic request-state cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dart.verification import synthetic
from dart.verification.reduction import ReductionResult, failure_key, reduce_failure
from dart.verification.state_sequences import StateOperation, StateSequenceCheck

MAX_REPORT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class StateReplay:
    seed: int
    operations: tuple[StateOperation, ...]
    source_sha256: str
    selection: str


def load_state_replay(path: str | Path) -> StateReplay:
    """Read a local report; only the fixed tiny CPU model is supported.

    Saved results are not trusted as current evidence. The caller must execute
    the selected operations again; versions and failures can change after a fix.
    """
    with Path(path).open("rb") as source:
        raw = source.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("synthetic replay report exceeds 1 MiB")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("schema") != "dart-synthetic-state-check-v1":
            raise ValueError("expected a synthetic state report")
        seed = data["seed"]
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("invalid synthetic replay seed")
        model = {"architecture": "LlamaForCausalLM", "config": synthetic.synthetic_model_config(),
                 "device": "cpu", "dtype": "float32", "threads": 1}
        if data["model"] != model:
            raise ValueError("synthetic replay requires the fixed tiny CPU model")
        reduction = data.get("reduction")
        if reduction is not None and (not isinstance(reduction, dict)
                                      or type(reduction.get("reproduced")) is not bool):
            raise ValueError("invalid synthetic reduction record")
        selection = "reduced" if reduction is not None and reduction["reproduced"] else "original"
        if selection == "reduced":
            rows = reduction["reduced"]
        else:
            steps = data["steps"]
            if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
                raise ValueError("synthetic replay requires between 1 and 64 steps")
            rows = [step["operation"] for step in steps]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise ValueError("synthetic replay requires between 1 and 64 operations")
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("tokens", []), list)
                    or not isinstance(row.get("parents", []), list)):
                raise ValueError("invalid synthetic operation")
            if max(len(row.get("tokens", [])), len(row.get("parents", []))) > synthetic.CONTEXT_LENGTH + 1:
                raise ValueError("synthetic operation exceeds the token limit")
        operations = synthetic._bounded_state_operations(tuple(StateOperation(**row) for row in rows))
    except (KeyError, TypeError, RecursionError) as error:
        raise ValueError("invalid synthetic replay report") from error
    return StateReplay(seed, operations, hashlib.sha256(raw).hexdigest(), selection)


def reduce_state_check(report: StateSequenceCheck, seed: int, *, max_evaluations: int = 32) -> ReductionResult | None:
    """Use the report's seed; count it as the first check and replay each trial fresh."""
    if type(max_evaluations) is not int or not 2 <= max_evaluations <= 32:
        raise ValueError("synthetic reduction needs between 2 and 32 evaluations")
    synthetic._generator(seed)
    operations = synthetic._bounded_state_operations(tuple(step.operation for step in report.steps))
    initial_failure = failure_key(report)
    if initial_failure is None:
        return None
    initial = True

    def evaluate(candidate):
        nonlocal initial
        if initial:
            initial = False
            return initial_failure  # The supplied report already checked this case.
        return failure_key(synthetic.run_synthetic_state_check(seed, operations=candidate))

    return reduce_failure(operations, evaluate, max_evaluations=max_evaluations)
