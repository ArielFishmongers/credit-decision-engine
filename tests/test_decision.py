"""The decision layer: the sweep's arithmetic, and claim 3 as a property.

Built on synthetic applicant frames rather than the real book, so the right answer is
known by construction. The sweep is pure arithmetic over a sorted vector — exactly the
kind of code that returns plausible numbers when it is subtly wrong, which is why the
identities at full and zero approval are pinned rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cde.config import Config, load_config
from cde.economics.decision import evaluate, rules_table, sweep

RARE_RATE = 0.06


@pytest.fixture
def config() -> Config:
    return load_config()


def applicants(n: int = 20_000, seed: int = 4, rate: float = RARE_RATE) -> pd.DataFrame:
    """A book whose risk ranking works and whose bad loans lose far more than good ones earn.

    That asymmetry is the whole economics of lending — here a default costs about thirty
    times what a performing loan earns — and it is what makes the profit-maximising
    cutoff differ from any symmetric statistical one.
    """
    rng = np.random.default_rng(seed)
    score = rng.beta(1.4, 18.0, n)
    defaulted = rng.random(n) < np.clip(score * (rate / score.mean()), 0.0, 1.0)
    realised = np.where(defaulted, -30_000.0, 1_000.0) + rng.normal(0.0, 400.0, n)
    return pd.DataFrame(
        {
            "loan_identifier": [f"L{i:06d}" for i in range(n)],
            "score": score,
            # Expected NPV that is broadly right about ordering: positive for low scores.
            "expected_npv": 1_000.0 - 31_000.0 * score,
            "realised_npv": realised,
            "resolved": True,
            "defaulted": defaulted,
        }
    )


def test_full_approval_reproduces_the_whole_book(config: Config) -> None:
    """The identity that pins the cumulative arithmetic."""
    frame = applicants()
    curve = sweep(config, frame)
    last = curve.iloc[-1]
    assert last["approval_rate"] == pytest.approx(1.0)
    assert last["npv_total"] == pytest.approx(frame["realised_npv"].sum())
    assert last["npv_per_applicant"] == pytest.approx(frame["realised_npv"].mean())
    # Declining nobody means no positive predictions, so recall and F1 are zero...
    assert last["recall"] == pytest.approx(0.0)
    assert last["f1"] == pytest.approx(0.0)
    # ...while accuracy is the share of loans that did NOT default, which on rare-event
    # data is close to one. This is the whole problem with accuracy in one assertion.
    assert last["accuracy"] == pytest.approx(1.0 - frame["defaulted"].mean())
    assert last["accuracy"] > 0.9


def test_zero_approval_earns_and_loses_nothing(config: Config) -> None:
    curve = sweep(config, applicants())
    first = curve.iloc[0]
    assert first["approval_rate"] == pytest.approx(0.0)
    assert first["npv_total"] == pytest.approx(0.0)
    # Declining everybody catches every default, so recall is perfect and precision is
    # the base rate. A rule can always have perfect recall; it is worth nothing.
    assert first["recall"] == pytest.approx(1.0)
    assert first["precision"] == pytest.approx(RARE_RATE, abs=0.01)


def test_npv_per_applicant_divides_by_every_applicant(config: Config) -> None:
    """Not by the approved subset. Dividing by the approved book would rank a tiny
    immaculate portfolio above a large profitable one, because it ignores that the
    declining lender has not deployed the capital."""
    frame = applicants()
    curve = sweep(config, frame)
    half = curve.iloc[len(curve) // 2]
    approved = int(half["approved"])
    by_applicant = half["npv_total"] / len(frame)
    assert half["npv_per_applicant"] == pytest.approx(by_applicant)
    # And it is NOT the mean over the approved loans, which is a larger number here.
    assert half["npv_per_applicant"] != pytest.approx(half["npv_total"] / approved)


def test_accuracy_is_maximised_by_approving_everyone(config: Config) -> None:
    """**Claim 3's negative half, as arithmetic rather than an anecdote.**

    Accuracy is (true negatives + true positives) / n. On rare-event data the true
    negatives dominate, so every applicant declined trades a certain true negative for a
    probable false positive, and accuracy falls monotonically as approval falls. The
    accuracy-maximising rule is therefore the do-nothing rule — and it reports a headline
    in the nineties while doing it.
    """
    curve = sweep(config, applicants())
    best = curve.iloc[int(curve["accuracy"].to_numpy().argmax())]
    assert best["approval_rate"] == pytest.approx(1.0)
    assert best["accuracy"] > 0.9
    # Monotone in approval rate, up to the sweep's granularity.
    accuracy = curve["accuracy"].to_numpy()
    assert np.all(np.diff(accuracy) > -1e-9)


def test_the_profit_maximising_cutoff_declines_a_real_share(config: Config) -> None:
    """Claim 3's positive half: profit peaks strictly inside the range.

    With a default costing thirty times what a good loan earns, there is an interior
    optimum, and it is nowhere near where accuracy or F1 peak. If profit also peaked at
    full approval the whole decision layer would be pointless — so this is the test that
    the experiment has any content at all.
    """
    curve = sweep(config, applicants())
    best_index = int(curve["npv_per_applicant"].to_numpy().argmax())
    best = curve.iloc[best_index]
    assert 0.05 < best["approval_rate"] < 0.95
    # Strictly better than approving everyone, and than approving nobody.
    assert best["npv_per_applicant"] > curve.iloc[-1]["npv_per_applicant"]
    assert best["npv_per_applicant"] > 0.0
    # And it is a different cutoff from the one accuracy would choose.
    accuracy_best = curve.iloc[int(curve["accuracy"].to_numpy().argmax())]
    assert best["approval_rate"] < accuracy_best["approval_rate"] - 0.05


def test_every_rule_is_located_and_approve_all_is_the_full_book(config: Config) -> None:
    frame = applicants()
    result = evaluate(config, frame)
    rules = result.rules.set_index("rule")
    assert rules.loc["approve all", "approval_rate"] == pytest.approx(1.0)
    assert rules.loc["approve all", "npv_per_applicant"] == pytest.approx(
        frame["realised_npv"].mean()
    )
    # The oracle is by construction the best point on the curve, so nothing can beat it.
    oracle = rules.loc["oracle (hindsight)", "npv_per_applicant"]
    assert (rules["npv_per_applicant"] <= oracle + 1e-6).all()
    assert "expected NPV > 0" in rules.index


def test_thresholds_chosen_elsewhere_are_carried_across_as_thresholds(
    config: Config,
) -> None:
    """Not as approval rates.

    The statistical cutoffs are chosen on the in-time holdout and applied to the test
    book. The rule IS the score threshold; the two books have different score
    distributions, so carrying the chosen *rate* across would silently redefine it.
    """
    test = applicants(seed=1)
    # A holdout whose scores run systematically lower, so rate and threshold disagree.
    holdout = applicants(seed=2)
    holdout["score"] = holdout["score"] * 0.5
    curve = sweep(config, test)
    picked = rules_table(config, test, curve, sweep(config, holdout))
    naive = rules_table(config, test, curve, None)
    f1_picked = picked.set_index("rule").loc["F1-max", "approval_rate"]
    f1_naive = naive.set_index("rule").loc["F1-max", "approval_rate"]
    # Choosing on a shifted holdout must land somewhere different from choosing on the
    # test set itself; if it did not, the chooser is being ignored.
    assert f1_picked != pytest.approx(f1_naive)


def test_unresolved_applicants_are_excluded_from_the_statistical_metrics(
    config: Config,
) -> None:
    """A loan censored before the binary horizon cannot answer "did it default within 36
    months". Counting it as a non-default understates the rate by the share of the book
    that stopped reporting."""
    frame = applicants(n=2_000, seed=6)
    frame.loc[frame.index[:500], "resolved"] = False
    frame.loc[frame.index[:500], "defaulted"] = False
    curve = sweep(config, frame)
    # Accuracy is measured over the 1,500 resolved loans only, so it must match the
    # resolved base rate at full approval rather than the diluted one.
    resolved = frame[frame["resolved"]]
    assert curve.iloc[-1]["accuracy"] == pytest.approx(1.0 - resolved["defaulted"].mean())
    # Realised NPV still counts everybody: the cash is real whether or not the binary
    # target is readable.
    assert curve.iloc[-1]["npv_total"] == pytest.approx(frame["realised_npv"].sum())
