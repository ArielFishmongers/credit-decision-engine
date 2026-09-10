"""Stage 1 vintage curves: hazards, survival and cumulative incidence by cohort.

The SQL in ``sql/vintage_curves.sql`` carries the derivations and the argument for
why one of the three cumulative estimators is right and the others are not. This
module runs it, pulls the result into a DataFrame, and draws the charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template

import duckdb
import pandas as pd

from cde.config import Config


@dataclass(frozen=True)
class VintageSummary:
    """Cumulative outcomes at the horizon, per vintage. The headline table."""

    frame: pd.DataFrame

    def summary(self) -> str:
        lines = [
            f"{'vintage':>8} {'loans':>8} {'default %':>10} {'naive %':>9} "
            f"{'1-KM %':>8} {'prepaid %':>10} {'alive %':>8}",
        ]
        for row in self.frame.itertuples():
            lines.append(
                f"{row.vintage:>8} {row.cohort_size:>8,} {row.cif_default * 100:>10.2f} "
                f"{row.naive_default * 100:>9.2f} {row.km_default * 100:>8.2f} "
                f"{row.cif_prepay * 100:>10.2f} {row.survival * 100:>8.2f}"
            )
        return "\n".join(lines)


def _sql(config: Config) -> str:
    template = Template(
        resources.files("cde.eval.sql").joinpath("vintage_curves.sql").read_text()
    )
    return template.substitute(horizon=config.horizon.max_loan_age_months)


def build(config: Config, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create the vintage views and return the full hazard/incidence table."""
    con.execute(_sql(config))
    return con.execute("SELECT * FROM vintage_hazards ORDER BY vintage, loan_age").df()


def summarise(curves: pd.DataFrame, horizon: int) -> VintageSummary:
    """Cumulative outcomes at the horizon, one row per vintage."""
    at_horizon = (
        curves[curves["loan_age"] == curves.groupby("vintage")["loan_age"].transform("max")]
        .sort_values("vintage")
        .reset_index(drop=True)
    )
    return VintageSummary(at_horizon)


def train_test_split(config: Config, curves: pd.DataFrame) -> pd.DataFrame:
    """Label each vintage train or test, for consistent colouring across charts."""
    labels = {v: "train" for v in config.data.train_vintages}
    labels.update({v: "test" for v in config.data.test_vintages})
    out = curves.copy()
    out["split"] = out["vintage"].map(labels)
    return out


def write_curves(curves: pd.DataFrame, path: Path) -> Path:
    """Persist the curve table so the charts and the write-up cite the same numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(path, index=False)
    return path
