"""Unit tests for the diffusion stack: canvas, schedule, denoise,
reconciliation slot accounting, and the training losses."""

from __future__ import annotations

import pytest
import torch

from dart.diffusion.canvas import Canvas
from dart.diffusion.denoise import denoise
from dart.diffusion.noise import (
    cosine_mask_schedule,
    mask_corrupt,
    renoise_suffix,
    span_permute_corrupt,
    syntax_aware_corrupt,
)
from dart.diffusion.reconciliation import ReconcilePolicy, make_warm_canvas
from dart.diffusion.training_losses import LossWeights, dart_drafter_loss
from dart.runtime.types import VerifyResult

V = 50


def test_cosine_schedule_terminates_and_decreases():
    for steps in (1, 2, 4, 16):
        sched = cosine_mask_schedule(steps)
        assert len(sched) == steps
        assert sched[-1] == pytest.approx(0.0, abs=1e-9)
        assert all(a >= b for a, b in zip(sched, sched[1:]))


def test_canvas_pad_slice_roundtrip():
    c = Canvas.from_tokens([1, 2, 3], confidence=[0.9, 0.8, 0.7])
    p = c.padded_to(6)
    assert p.length == 6 and p.num_masked == 3
    s = p.sliced(1, 4)
    assert s.tokens[:2].tolist() == [2, 3]
    assert s.masked.tolist() == [False, False, True]
    with pytest.raises(ValueError):
        p.as_draft()  # masked slots present


def test_mask_corrupt_extremes():
    t = torch.arange(100)
    all_mask, _ = mask_corrupt(t, 1.0)
    no_mask, _ = mask_corrupt(t, 0.0)
    assert bool(all_mask.all()) and not bool(no_mask.any())


def test_span_permute_is_a_span_reordering():
    g = torch.Generator().manual_seed(3)
    t = torch.arange(12)
    out, moved = span_permute_corrupt(t, span_len=4, generator=g)
    # Output is a permutation of the same multiset of tokens...
    assert sorted(out.tolist()) == sorted(t.tolist())
    # ...assembled from intact length-4 spans (local order preserved).
    spans = {tuple(t[i : i + 4].tolist()) for i in range(0, 12, 4)}
    out_spans = {tuple(out[i : i + 4].tolist()) for i in range(0, 12, 4)}
    assert out_spans == spans
    assert moved.dtype == torch.bool and moved.shape == t.shape
    # Fewer than 2 spans: nothing to permute.
    out1, moved1 = span_permute_corrupt(torch.arange(3), span_len=4)
    assert out1.tolist() == [0, 1, 2] and not bool(moved1.any())


def test_syntax_aware_corrupt_never_masks_protected_tokens():
    t = torch.tensor([5, 1, 6, 2, 7, 1])  # ids 1,2 are "syntax"
    g = torch.Generator().manual_seed(0)
    mask = syntax_aware_corrupt(t, 1.0, torch.tensor([1, 2]), generator=g)
    protected = torch.isin(t, torch.tensor([1, 2]))
    assert not bool(mask[protected].any())  # structural tokens survive
    assert bool(mask[~protected].all())  # everything else masked at level 1.0


def _uniformish_model(vocab=V, peak=None, conf=0.99):
    """Model fn emitting near-uniform logits, or peaked on `peak`."""

    def fn(ctx, canvas):
        b = canvas.length
        logits = torch.zeros(b, vocab)
        if peak is not None:
            logits[:, peak] = torch.log(
                torch.tensor(conf * (vocab - 1) / (1 - conf))
            )
        return logits

    return fn


def test_denoise_fills_everything_within_budget():
    canvas = Canvas.empty(12)
    stats = denoise(_uniformish_model(), torch.tensor([1, 2, 3]), canvas, num_steps=4)
    assert canvas.num_masked == 0
    assert 1 <= stats.steps_used <= 4
    assert canvas.as_draft() == canvas.tokens.tolist()


def test_denoise_adaptive_stop_single_step_when_confident():
    canvas = Canvas.empty(12)
    stats = denoise(
        _uniformish_model(peak=7), torch.tensor([1]), canvas,
        num_steps=8, min_confidence=0.65, adaptive_stop=True,
    )
    assert stats.steps_used == 1  # everything confident on step one
    assert canvas.as_draft() == [7] * 12


def test_denoise_respects_prefilled_anchors():
    canvas = Canvas.empty(6)
    canvas.fill(torch.tensor([0, 3]), torch.tensor([41, 42]), torch.tensor([1.0, 1.0]))
    denoise(_uniformish_model(peak=7), torch.tensor([1]), canvas, num_steps=3)
    out = canvas.as_draft()
    assert out[0] == 41 and out[3] == 42  # anchors never re-predicted
    assert all(t == 7 for i, t in enumerate(out) if i not in (0, 3))


def test_renoise_suffix_thresholds_and_cap():
    c = Canvas.from_tokens(
        [10, 11, 12, 13], confidence=[0.9, 0.2, 0.95, 0.4]
    )
    out = renoise_suffix(c, keep_confidence_above=0.5, max_keep_fraction=1.0)
    assert out.masked.tolist() == [False, True, False, True]
    capped = renoise_suffix(c, keep_confidence_above=0.0, max_keep_fraction=0.5)
    assert capped.num_masked >= 2  # cap forces extra re-masking


def _vr(committed, proposed, accepted, all_acc=False, finished=False):
    return VerifyResult(
        committed=committed, n_draft_proposed=proposed,
        n_draft_accepted=accepted, all_accepted=all_acc, finished=finished,
    )


def test_make_warm_canvas_slot_accounting():
    last = Canvas.from_tokens(
        [20, 21, 22, 23, 24, 25], confidence=[0.9, 0.9, 0.9, 0.1, 0.9, 0.1]
    )
    policy = ReconcilePolicy(mode="reconcile", keep_confidence_above=0.5,
                             max_keep_fraction=1.0)
    # 2 drafts accepted, correction replaced slot 2 -> suffix is slots 3..5.
    warm = make_warm_canvas(last, _vr([20, 21, 99], 6, 2), policy)
    assert warm is not None and warm.length == 3
    assert warm.tokens.tolist() == [23, 24, 25]
    assert warm.masked.tolist() == [True, False, True]  # by confidence


def test_make_warm_canvas_none_cases():
    last = Canvas.from_tokens([1, 2, 3])
    pol = ReconcilePolicy(mode="reconcile")
    assert make_warm_canvas(None, _vr([9], 3, 0), pol) is None
    assert make_warm_canvas(last, _vr([1, 2, 3, 9], 3, 3, all_acc=True), pol) is None
    assert make_warm_canvas(last, _vr([1], 3, 0, finished=True), pol) is None
    assert (
        make_warm_canvas(last, _vr([1, 2, 9], 3, 2), pol) is None
    )  # suffix empty: rejection at the last slot
    assert make_warm_canvas(last, _vr([9], 3, 0), ReconcilePolicy(mode="discard")) is None


def test_dart_drafter_loss_shapes_and_padding_invariance():
    torch.manual_seed(0)
    n, b = 3, 8
    logits = torch.randn(n, b, V, requires_grad=True)
    x0 = torch.randint(0, V, (n, b))
    masked = torch.rand(n, b) < 0.5
    masked[:, 0] = True  # ensure some supervision
    tlogits = torch.randn(n, b, V)
    valid = torch.ones(n, b, dtype=torch.bool)
    valid[:, -2:] = False

    parts = dart_drafter_loss(logits, x0, masked, tlogits, LossWeights(), valid)
    assert all(torch.isfinite(v) for v in parts.values())
    parts["total"].backward()

    # Corrupting the *invalid* region must not change any loss term.
    x0_c = x0.clone()
    x0_c[:, -2:] = 0
    t_c = tlogits.clone()
    t_c[:, -2:] = 123.0
    parts2 = dart_drafter_loss(logits.detach(), x0_c, masked, t_c, LossWeights(), valid)
    for k in ("denoise", "kl", "accept", "calibration", "total"):
        assert parts2[k].item() == pytest.approx(parts[k].item(), rel=1e-5)
