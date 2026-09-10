"""The ablation's threshold arithmetic, and the degeneracy that is claim 3's negative half.

`accuracy_maximising_threshold` decides what B1 is, so if it silently returned "approve
everything" for the wrong reason the ablation's most quoted row would be an artefact of a
bug rather than a property of the data. It is tested both ways: degenerate where the event
is rare, interior where it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from cde.eval.harness import ARMS, accuracy_maximising_threshold


def ranked(n: int = 20_000, rate: float = 0.02, seed: int = 3):
    """Scores that genuinely rank, with a realised event rate close to ``rate``.

    ``score`` is uniform on [0, 1] and the default probability is ``2 * rate * score``,
    whose mean is exactly ``rate`` — so the realised base rate is the one asked for and
    an assertion about it means what it says. An earlier version multiplied by a
    free "strength" factor and delivered 6.25% when asked for 2%.
    """
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.0, 1.0, n)
    defaulted = rng.random(n) < np.clip(2.0 * rate * score, 0.0, 1.0)
    return score, defaulted, np.ones(n, dtype=bool)


def test_on_rare_events_accuracy_is_maximised_by_declining_nobody() -> None:
    """**The finding, not a defect.**

    Accuracy is ``(TN + TP) / n``. At a low base rate the true negatives dominate, so
    every applicant declined trades a certain true negative for a probable false
    positive. The maximum therefore sits at full approval, and the arm reports a headline
    in the high nineties while making no decisions at all.
    """
    score, defaulted, resolved = ranked(rate=0.02)
    threshold, accuracy = accuracy_maximising_threshold(score, defaulted, resolved)
    assert (score < threshold).all()
    # The identity that holds whenever the optimum is full approval: every resolved
    # non-default is a true negative and every default a false negative, so accuracy is
    # exactly one minus the base rate. Asserting the identity rather than a round number
    # means the test cannot pass for the wrong reason.
    assert accuracy == pytest.approx(1.0 - defaulted.mean())
    assert accuracy > 0.9


def test_the_threshold_admits_exactly_the_intended_loans() -> None:
    """Returned for use as ``approve if score < threshold``, so it must sit strictly
    above the last approved score — an off-by-one here would silently move the cutoff by
    one loan on every arm."""
    score, defaulted, resolved = ranked(rate=0.02)
    threshold, _ = accuracy_maximising_threshold(score, defaulted, resolved)
    assert threshold > score.max()
    assert np.isfinite(threshold)


def test_with_a_common_event_the_threshold_is_interior() -> None:
    """The control that shows the function is not simply always returning the maximum.

    At a 40% event rate declining the worst applicants does improve accuracy, so the
    optimum moves inside the range. The degeneracy above is a property of rare events,
    which is the whole point.
    """
    score, defaulted, resolved = ranked(rate=0.40, seed=11)
    assert 0.2 < defaulted.mean() < 0.7
    threshold, accuracy = accuracy_maximising_threshold(score, defaulted, resolved)
    approved = float((score < threshold).mean())
    assert 0.05 < approved < 0.99, approved
    assert accuracy > 1.0 - defaulted.mean()


def test_only_resolved_loans_inform_the_threshold() -> None:
    """An unresolved loan has no readable outcome; counting it as a non-default would
    drag the threshold toward approving more."""
    score, defaulted, resolved = ranked(rate=0.30, seed=5)
    partial = resolved.copy()
    partial[: len(partial) // 2] = False
    # Corrupt the outcomes of the unresolved half; the answer must not move.
    corrupted = defaulted.copy()
    corrupted[: len(partial) // 2] = True
    a, _ = accuracy_maximising_threshold(score, defaulted, partial)
    b, _ = accuracy_maximising_threshold(score, corrupted, partial)
    assert a == pytest.approx(b)


def test_no_resolved_loans_is_refused() -> None:
    score, defaulted, _ = ranked(rate=0.02)
    with pytest.raises(ValueError, match="no resolved loans"):
        accuracy_maximising_threshold(score, defaulted, np.zeros(len(score), dtype=bool))


def test_the_arms_are_declared_in_order_and_described() -> None:
    """The table's row order is the ablation's logic: each arm adds one thing to the one
    above it, so the steps decompose the total only if the order is right."""
    assert list(ARMS) == ["B0", "B1", "B2", "B3", "B4a", "B4b"]
    for key, (label, isolates) in ARMS.items():
        assert label and isolates, key
