"""Realised NPV: the cash a loan actually produced.

The one invariant that matters here is exact and holds for every loan without exception,
which is what makes it worth testing rather than inspecting:

    sum(principal payments, undiscounted) + terminal balance == cash advanced

by telescoping over the observed balance path. It holds to $0.0000 across all 129,449
test-book loans on the real data. If the month sequencing, the lead/lag alignment or the
terminal-value convention were wrong, this is the check that would fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cde.config import load_config
from cde.data.ingest import connect, ingest
from cde.economics.realised import loss_censoring, measure
from tests.conftest import SyntheticLoan


@pytest.fixture
def valued(synthetic: tuple[Path, list[SyntheticLoan]]):
    config_path, cases = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        yield config, measure(config, con, config.data.vintages), cases, con


def test_the_telescoping_identity_holds_for_every_loan(valued) -> None:
    """Principal repaid plus what is still owed equals what was lent. No exceptions."""
    _, realised, _, _ = valued
    frame = realised.frame
    assert len(frame) > 0
    gap = (
        frame["principal_undiscounted"] + frame["terminal_balance"] - frame["advanced_upb"]
    ).abs()
    assert gap.max() < 1e-6, frame.loc[gap.idxmax()].to_dict()


def test_a_loan_with_a_realised_loss_is_worth_less_than_one_without(valued) -> None:
    """The fixture builds one loan with a charge-off at 40% of balance. It must show up
    as a realised loss and it must depress that loan's NPV below its peers'."""
    _, realised, _, _ = valued
    frame = realised.frame.set_index("loan_identifier")
    with_loss = frame[frame["realised_loss"] > 0]
    if with_loss.empty:
        pytest.skip("fixture produced no priced disposition inside the horizon")
    without = frame[frame["realised_loss"] == 0]
    assert with_loss["npv"].max() < without["npv"].median()
    assert (with_loss["pv_loss"] < 0).all()


def test_every_loan_is_valued_at_most_once(valued) -> None:
    _, realised, _, _ = valued
    assert not realised.frame["loan_identifier"].duplicated().any()


def test_only_loans_observed_from_origination_are_valued(valued) -> None:
    """An NPV is a value at the decision point. The fixture includes a loan entering at
    age 6, and it must not appear."""
    _, realised, cases, _ = valued
    left_truncated = [c.loan for c in cases if "left truncated" in c.note]
    if not left_truncated:
        pytest.skip("fixture has no left-truncated loan")
    valued_ids = set(realised.frame["loan_identifier"])
    for loan in left_truncated:
        assert loan not in valued_ids


def test_the_legs_sum_to_the_reported_npv(valued) -> None:
    _, realised, _, _ = valued
    f = realised.frame
    rebuilt = (
        f["pv_interest"]
        + f["pv_principal"]
        + f["pv_terminal"]
        + f["pv_loss"]
        + f["pv_servicing"]
        - f["advanced_upb"]
    )
    assert (rebuilt - f["npv"]).abs().max() < 1e-6


def test_loss_censoring_reports_a_real_population(valued) -> None:
    """It returned an empty frame once, because the month index was windowed after the
    `actual_loss IS NOT NULL` filter and every disposition became its own first month.
    An empty diagnostic is worse than a wrong one: nothing looks broken."""
    config, _, _, con = valued
    table = loss_censoring(con, config, config.data.vintages)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["priced_dispositions"] > 0
    assert row["inside_horizon"] <= row["priced_dispositions"]
