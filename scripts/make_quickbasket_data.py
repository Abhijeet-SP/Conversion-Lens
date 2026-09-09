"""Generate the two QuickBasket A/B-test datasets.

QuickBasket is a fictional 10-minute grocery delivery service. A ~=499 free-delivery
threshold already exists in both arms. The experiment adds a progress nudge on the cart
page for the treatment arm only:

    control    flat ~=35 delivery fee shown, no nudge
    treatment  "Add ~=120 more to unlock FREE delivery"

Randomisation is one cart session per user, so every observation is independent.

Generative model (this is the ground truth the notebooks must recover)
---------------------------------------------------------------------
order_placed ~ Bernoulli(p) with
    p = base[city_tier] + new_user_adjustment + 0.004 * (cart_items - mean_items)
        + conversion_effect[city_tier] * is_treatment

    base               metro 0.360   tier_2 0.340   tier_3 0.310
    conversion_effect  metro -0.004  tier_2 -0.010  tier_3 -0.050

basket_value_inr starts from the pre-nudge cart_subtotal_inr (identical in both arms) and
the treatment transformation is applied only to treatment sessions:

    1. threshold pull  a subtotal in [300, 499) is pushed to Uniform(500, 600) with
       probability PULL_PROB[city_tier] -- the nudge doing its job
    2. residual lift   everything is then scaled by (1 + RESIDUAL_LIFT[city_tier])

Both steps are scaled by a novelty factor that decays over the four experiment weeks
(week 1 is strongest), so the pooled lift overstates the steady-state lift.

Deliberate data-quality defects, for the EDA notebooks to find
-------------------------------------------------------------
* ~1.2% duplicated session_id rows (double-fired analytics event)
* 250 sessions with order_placed = 1 but no matching row in the orders table
* ~0.3% of orders with a missing basket_value_inr
* 12 bulk orders above ~=40,000 (6 per arm, so they inflate variance without biasing)
* device recorded as both "android" and "Android"
* web traffic splits 48.5/51.5 instead of 50/50 -- a real sample-ratio mismatch

Run:  python3 scripts/make_quickbasket_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_SESSIONS = 160_000
N_LOYAL_USERS = 3_200

EXPERIMENT_START = pd.Timestamp("2026-02-02")
EXPERIMENT_WEEKS = 4
PRE_PERIOD_START = pd.Timestamp("2026-01-19")
PRE_PERIOD_DAYS = 14

FREE_DELIVERY_THRESHOLD = 499.0
FLAT_DELIVERY_FEE = 35.0

TIERS = ["metro", "tier_2", "tier_3"]
TIER_SHARE = [0.40, 0.35, 0.25]
DEVICES = ["android", "ios", "web"]
DEVICE_SHARE = [0.62, 0.23, 0.15]

BASE_CONVERSION = {"metro": 0.360, "tier_2": 0.340, "tier_3": 0.310}
CONVERSION_EFFECT = {"metro": -0.004, "tier_2": -0.010, "tier_3": -0.050}
PRICE_FACTOR = {"metro": 1.10, "tier_2": 0.94, "tier_3": 0.80}
PULL_PROB = {"metro": 0.62, "tier_2": 0.45, "tier_3": 0.18}
RESIDUAL_LIFT = {"metro": 0.105, "tier_2": 0.075, "tier_3": 0.035}
DELIVERY_COST = {"metro": (38.0, 6.0), "tier_2": (44.0, 7.0), "tier_3": (60.0, 9.0)}

# Week-1 novelty, decaying to below-average by week 4.
NOVELTY = np.array([1.35, 1.15, 0.95, 0.80])

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
SESSIONS_PATH = ROOT / "data" / "qb_cart_sessions.csv"
ORDERS_PATH = ROOT / "data" / "qb_orders.csv"


def draw_timestamps(rng: np.random.Generator, n: int, start: pd.Timestamp, days: int):
    """Weekend-heavy days, evening-peak hours -- a realistic delivery volume curve."""
    day_weight = np.array([0.12, 0.12, 0.13, 0.14, 0.16, 0.17, 0.16])
    day_weight = np.repeat(day_weight, days // 7 + 1)[:days]
    day_weight = day_weight / day_weight.sum()
    day = rng.choice(days, size=n, p=day_weight)

    hour_weight = np.array(
        [0.2, 0.1, 0.1, 0.1, 0.2, 0.6, 1.4, 2.2, 3.0, 3.4, 3.6, 4.2,
         4.6, 3.8, 3.0, 2.8, 3.4, 4.8, 6.4, 7.2, 6.4, 4.6, 2.8, 1.2]
    )
    hour = rng.choice(24, size=n, p=hour_weight / hour_weight.sum())
    minute = rng.integers(0, 60, size=n)
    second = rng.integers(0, 60, size=n)
    return (
        start
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )


def build_sessions(rng: np.random.Generator) -> pd.DataFrame:
    n = N_SESSIONS
    city_tier = rng.choice(TIERS, size=n, p=TIER_SHARE)
    device = rng.choice(DEVICES, size=n, p=DEVICE_SHARE)

    # Sample-ratio mismatch: the web client mis-buckets a slice of its traffic.
    p_treatment = np.where(device == "web", 0.485, 0.500)
    group = np.where(rng.random(n) < p_treatment, "treatment", "control")

    is_new_user = (rng.random(n) < 0.28).astype(int)
    cart_items = np.clip(1 + rng.poisson(3.4, size=n), 1, 14)

    price_factor = np.vectorize(PRICE_FACTOR.get)(city_tier)
    cart_subtotal = cart_items * 95.0 * price_factor * rng.lognormal(0.0, 0.35, size=n)

    base = np.vectorize(BASE_CONVERSION.get)(city_tier)
    effect = np.vectorize(CONVERSION_EFFECT.get)(city_tier)
    p = (
        base
        + np.where(is_new_user == 1, -0.045, 0.0175)
        + 0.004 * (cart_items - cart_items.mean())
        + np.where(group == "treatment", effect, 0.0)
    )
    order_placed = (rng.random(n) < np.clip(p, 0.02, 0.95)).astype(int)

    sessions = pd.DataFrame(
        {
            "session_id": [f"S{i:07d}" for i in range(1, n + 1)],
            "user_id": [f"U{i:06d}" for i in rng.permutation(n) + 1],
            "group": group,
            "session_ts": draw_timestamps(rng, n, EXPERIMENT_START, EXPERIMENT_WEEKS * 7),
            "city_tier": city_tier,
            "device": device,
            "is_new_user": is_new_user,
            "cart_items": cart_items,
            "cart_subtotal_inr": cart_subtotal.round(2),
            "order_placed": order_placed,
        }
    )
    return sessions.sort_values("session_ts").reset_index(drop=True)


def apply_nudge(rng: np.random.Generator, subtotal, tier, treated, novelty):
    """Threshold pull + residual lift. Control baskets only get checkout-time noise."""
    basket = subtotal * rng.lognormal(0.0, 0.08, size=len(subtotal))

    pull_prob = np.vectorize(PULL_PROB.get)(tier) * novelty
    in_band = (basket >= 300.0) & (basket < FREE_DELIVERY_THRESHOLD)
    pulled = treated & in_band & (rng.random(len(basket)) < pull_prob)
    basket = np.where(pulled, rng.uniform(500.0, 600.0, size=len(basket)), basket)

    lift = np.vectorize(RESIDUAL_LIFT.get)(tier) * novelty
    return np.where(treated, basket * (1.0 + lift), basket)


def build_post_orders(rng: np.random.Generator, sessions: pd.DataFrame) -> pd.DataFrame:
    converted = sessions[sessions["order_placed"] == 1].reset_index(drop=True)
    week = ((converted["session_ts"] - EXPERIMENT_START).dt.days // 7).clip(0, 3).to_numpy()

    basket = apply_nudge(
        rng,
        converted["cart_subtotal_inr"].to_numpy(),
        converted["city_tier"].to_numpy(),
        (converted["group"] == "treatment").to_numpy(),
        NOVELTY[week],
    )

    return pd.DataFrame(
        {
            "session_id": converted["session_id"],
            "user_id": converted["user_id"],
            "group": converted["group"],
            "city_tier": converted["city_tier"],
            "phase": "post",
            "order_ts": converted["session_ts"] + pd.to_timedelta(rng.integers(30, 400, len(converted)), unit="s"),
            "items_count": converted["cart_items"],
            "basket_value_inr": basket,
        }
    )


def build_pre_orders(rng: np.random.Generator, sessions: pd.DataFrame) -> pd.DataFrame:
    """Order history for a loyal cohort, generated before launch -- no treatment effect.

    This is what makes the paired t-test (and its control-arm placebo) possible.
    """
    loyal = sessions.sample(N_LOYAL_USERS, random_state=SEED)[["user_id", "group", "city_tier"]]
    counts = rng.integers(3, 9, size=len(loyal))

    rows = loyal.loc[loyal.index.repeat(counts)].reset_index(drop=True)
    n = len(rows)
    items = np.clip(1 + rng.poisson(3.4, size=n), 1, 14)
    price_factor = np.vectorize(PRICE_FACTOR.get)(rows["city_tier"].to_numpy())
    subtotal = items * 95.0 * price_factor * rng.lognormal(0.0, 0.35, size=n)

    # Pre-period is history, so nobody is treated -- both arms share one distribution.
    basket = apply_nudge(rng, subtotal, rows["city_tier"].to_numpy(), np.zeros(n, dtype=bool), 1.0)

    rows["phase"] = "pre"
    rows["session_id"] = pd.NA
    rows["order_ts"] = draw_timestamps(rng, n, PRE_PERIOD_START, PRE_PERIOD_DAYS)
    rows["items_count"] = items
    rows["basket_value_inr"] = basket
    return rows


def add_economics(rng: np.random.Generator, orders: pd.DataFrame) -> pd.DataFrame:
    basket = orders["basket_value_inr"].to_numpy()
    unlocked = (basket >= FREE_DELIVERY_THRESHOLD).astype(int)

    mean, sd = zip(*[DELIVERY_COST[t] for t in orders["city_tier"]])
    cost = np.clip(rng.normal(mean, sd), 18.0, None)
    margin_rate = np.clip(rng.normal(0.18, 0.02, size=len(orders)), 0.10, 0.28)

    orders["free_delivery_unlocked"] = unlocked
    orders["delivery_fee_charged_inr"] = np.where(unlocked == 1, 0.0, FLAT_DELIVERY_FEE)
    orders["delivery_cost_inr"] = cost.round(2)
    orders["gross_margin_inr"] = (margin_rate * basket).round(2)
    orders["basket_value_inr"] = basket.round(2)
    return orders


def add_defects(rng: np.random.Generator, sessions: pd.DataFrame, orders: pd.DataFrame):
    # 12 bulk orders, split evenly so they add variance without biasing the comparison.
    post = orders.index[orders["phase"] == "post"]
    for arm in ("control", "treatment"):
        arm_rows = orders.loc[post][orders.loc[post, "group"] == arm]
        picked = rng.choice(arm_rows.index, size=6, replace=False)
        orders.loc[picked, "basket_value_inr"] *= rng.uniform(60, 90, size=6)
        orders.loc[picked, "gross_margin_inr"] = (0.18 * orders.loc[picked, "basket_value_inr"]).round(2)
        orders.loc[picked, "free_delivery_unlocked"] = 1
        orders.loc[picked, "delivery_fee_charged_inr"] = 0.0

    # 250 orders never landed in the orders pipeline, but the session says they converted.
    orphans = rng.choice(post, size=250, replace=False)
    orders = orders.drop(index=orphans)

    # ~0.3% of baskets failed to serialise.
    missing = rng.choice(orders.index, size=int(0.003 * len(orders)), replace=False)
    orders.loc[missing, "basket_value_inr"] = np.nan

    # Analytics double-fire: 1.2% of sessions logged twice.
    dupes = sessions.sample(frac=0.012, random_state=SEED)
    sessions = pd.concat([sessions, dupes]).sort_values("session_ts").reset_index(drop=True)

    # Inconsistent casing from one Android SDK version.
    android = sessions.index[sessions["device"] == "android"]
    sessions.loc[rng.choice(android, size=int(0.30 * len(android)), replace=False), "device"] = "Android"

    return sessions, orders


def self_check(sessions: pd.DataFrame, orders: pd.DataFrame) -> None:
    clean = sessions.drop_duplicates("session_id")
    assert len(clean) == N_SESSIONS, len(clean)
    assert clean["session_id"].is_unique and clean["user_id"].is_unique
    assert not clean[["group", "city_tier", "order_placed"]].isna().any().any()

    arms = clean.groupby("group")["order_placed"].agg(["size", "mean"])
    assert abs(arms.loc["control", "size"] - arms.loc["treatment", "size"]) < 1500
    conversion_effect = arms.loc["treatment", "mean"] - arms.loc["control", "mean"]
    assert -0.025 < conversion_effect < -0.010, conversion_effect

    post = orders[(orders["phase"] == "post") & orders["basket_value_inr"].notna()]
    aov = post.groupby("group")["basket_value_inr"].mean()
    assert aov["treatment"] > aov["control"], aov.to_dict()

    pre = orders[orders["phase"] == "pre"].groupby("group")["basket_value_inr"].mean()
    assert abs(pre["treatment"] - pre["control"]) < 15, pre.to_dict()

    print(f"sessions {len(sessions):,} rows ({len(clean):,} unique)   orders {len(orders):,} rows")
    print(f"conversion   control {arms.loc['control', 'mean']:.4f}   "
          f"treatment {arms.loc['treatment', 'mean']:.4f}   effect {conversion_effect * 100:+.2f} pp")
    print(f"AOV          control {aov['control']:>7.1f}   treatment {aov['treatment']:>7.1f}   "
          f"effect {aov['treatment'] - aov['control']:+.1f}")
    print(f"pre-period   control {pre['control']:>7.1f}   treatment {pre['treatment']:>7.1f}   "
          f"(placebo: should be flat)")


def main() -> None:
    rng = np.random.default_rng(SEED)
    sessions = build_sessions(rng)
    orders = pd.concat(
        [build_post_orders(rng, sessions), build_pre_orders(rng, sessions)],
        ignore_index=True,
    )
    orders = add_economics(rng, orders)
    sessions, orders = add_defects(rng, sessions, orders)

    orders = orders.sort_values(["phase", "order_ts"]).reset_index(drop=True)
    orders.insert(0, "order_id", [f"O{i:07d}" for i in range(1, len(orders) + 1)])
    orders = orders[
        ["order_id", "session_id", "user_id", "group", "city_tier", "phase", "order_ts",
         "items_count", "basket_value_inr", "free_delivery_unlocked",
         "delivery_fee_charged_inr", "delivery_cost_inr", "gross_margin_inr"]
    ]

    self_check(sessions, orders)
    sessions.to_csv(SESSIONS_PATH, index=False)
    orders.to_csv(ORDERS_PATH, index=False)
    print(f"wrote {SESSIONS_PATH.relative_to(ROOT)} and {ORDERS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
