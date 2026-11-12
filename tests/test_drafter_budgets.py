from __future__ import annotations

import pytest

from dart.drafters.compact_diffusion import CompactDiffusionDrafter
from dart.drafters.diffusion_small import DiffusionDrafter
from dart.drafters.diffusiongemma import RemoteDiffusionDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target, random_prompt


class Recorder:
    requires_entropy = False

    def __init__(self, name, uses_canvas=None):
        self.name, self.budgets = name, []
        if uses_canvas is not None:
            self.uses_canvas = uses_canvas

    def propose(self, committed, max_tokens):
        self.budgets.append(max_tokens)
        return []

    def observe(self, result):
        pass


@pytest.mark.parametrize("name, uses_canvas, expected", [
    ("lookup", None, 3), ("my_diffusion_variant", None, 3), ("lookup", True, 9), ("diffusion", False, 3),
])
def test_canvas_budget_follows_the_drafter_attribute(name, uses_canvas, expected):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = Recorder(name, uses_canvas)
    with DartEngine(target, drafter) as engine:
        engine.generate(random_prompt(81, 4), DecodeConfig(max_new_tokens=20, draft_len=3, canvas_len=9))
    assert drafter.budgets and max(drafter.budgets) == expected  # The last rounds shrink to the output budget.


@pytest.mark.parametrize("cls", [DiffusionDrafter, CompactDiffusionDrafter, RemoteDiffusionDrafter])
def test_built_in_canvas_drafters_declare_it(cls):
    assert cls.uses_canvas is True
