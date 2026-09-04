"""Reusable preparation and planning helpers for the conversion A/B test."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "ab_data.csv"
ALPHA = 0.05
POWER = 0.80
Z_ALPHA_TWO_SIDED = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
VALID_VARIANTS = {"control": "old_page", "treatment": "new_page"}


@dataclass(frozen=True)
class TestResult:
    control_n: int
    treatment_n: int
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    z_statistic: float
    p_value: float
    mde: float


def load_experiment(path: Path = DATA_PATH) -> list[dict[str, str]]:
    """Keep valid assignments and one (first) observation per user."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assigned = [row for row in rows if VALID_VARIANTS.get(row["group"]) == row["landing_page"]]
    seen: set[str] = set()
    return [row for row in assigned if not (row["user_id"] in seen or seen.add(row["user_id"]))]


def two_proportion_z_test(rows: list[dict[str, str]]) -> TestResult:
    """Two-sided, pooled two-proportion Z-test for the binary `converted` outcome."""
    total = Counter(row["group"] for row in rows)
    converted = Counter(row["group"] for row in rows if row["converted"] == "1")
    control_rate = converted["control"] / total["control"]
    treatment_rate = converted["treatment"] / total["treatment"]
    pooled_rate = sum(converted.values()) / sum(total.values())
    standard_error = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / total["control"] + 1 / total["treatment"]))
    z_statistic = (treatment_rate - control_rate) / standard_error
    p_value = math.erfc(abs(z_statistic) / math.sqrt(2))
    return TestResult(
        control_n=total["control"], treatment_n=total["treatment"],
        control_rate=control_rate, treatment_rate=treatment_rate,
        absolute_lift=treatment_rate - control_rate, z_statistic=z_statistic,
        p_value=p_value, mde=minimum_detectable_effect(control_rate, total["control"], total["treatment"]),
    )


def minimum_detectable_effect(baseline: float, control_n: int, treatment_n: int) -> float:
    """Absolute lift detectable with 5% two-sided alpha and 80% power."""
    def threshold(effect: float) -> float:
        null_se = math.sqrt(
            baseline * (1 - baseline) * (1 / control_n + 1 / treatment_n)
        )
        alternative_se = math.sqrt(
            baseline * (1 - baseline) / control_n
            + (baseline + effect) * (1 - baseline - effect) / treatment_n
        )
        return Z_ALPHA_TWO_SIDED * null_se + Z_POWER_80 * alternative_se

    low, high = 0.0, 1 - baseline
    for _ in range(60):
        middle = (low + high) / 2
        if middle >= threshold(middle):
            high = middle
        else:
            low = middle
    return high


def main() -> None:
    rows = load_experiment()
    result = two_proportion_z_test(rows)
    assert len(rows) == len({row["user_id"] for row in rows})
    print(result)


if __name__ == "__main__":
    main()
