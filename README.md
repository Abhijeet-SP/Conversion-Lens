# Conversion Lens

**Conversion fell 2.35 percentage points. We shipped anyway — to two of three city tiers.**

An end-to-end A/B testing case study for **QuickBasket**, a fictional 10-minute grocery
delivery service. One product change, measured two ways: a **z-test** on a binary outcome
and a **t-test** on a continuous one, with a guardrail metric that overturns the obvious
conclusion.

---

## The experiment

QuickBasket already has a ₹499 free-delivery threshold. The experiment adds a progress
nudge to the cart page for one arm only:

| Arm | Cart page |
| --- | --- |
| `control` | flat ₹35 delivery fee, no nudge |
| `treatment` | *"Add ₹120 more to unlock FREE delivery"* |

160,000 cart sessions, one per user, randomised 50/50, four weeks.

The nudge is supposed to trade a few orders for materially bigger baskets. Experiment 1
measures the orders it costs. Experiment 2 measures the baskets it buys — and whether the
margin survives the delivery QuickBasket gives away.

---

## Results

| Metric | Test | Effect | Verdict |
| --- | --- | --- | --- |
| Conversion rate | two-proportion **z-test** | **−2.35 pp** (34.38% → 32.03%), z = −9.98, p ≈ 2e-23 | significant **loss** |
| Revenue per session | Welch **t-test** | **+₹13.11** (+8.6%), t = +5.25, p ≈ 1.6e-07 | significant **gain** |
| Basket, loyal users | **paired t-test** | **+₹87.48**, t = 6.39, df = 519, p < 1e-09 | significant **gain** |
| Basket, control placebo | **paired t-test** | −₹11.58, p = 0.32 | null, exactly as required |
| **Margin per session** | Welch **t-test** | **+₹0.81**, 95% CI [−₹0.01, +₹1.63], p = 0.054 | **inconclusive** |

The guardrail is where the story turns. Pooled across all traffic the nudge moves a lot of
revenue and roughly no profit. Split by city tier, that pooled null resolves into three
different businesses:

| Segment | Margin per session | Bonferroni α = 0.0167 | Decision |
| --- | --- | --- | --- |
| metro | **+₹3.02**, p ≈ 0.0002 | significant | **ship** |
| tier_2 | +₹0.46, p ≈ 0.40 | not significant | **hold** — re-test at a ₹599 threshold |
| tier_3 | **−₹2.37**, p ≈ 0.0005 | significant | **do not ship** |

Annualised on the weeks 3–4 (post-novelty) estimates:

```
blanket rollout (all tiers)      ₹1,759,898   (p = 0.104  — cannot be defended)
metro-only rollout               ₹2,582,618   (p = 0.0006 — can)
tier-3 loss avoided by holding   ₹1,281,182
```

The segmented rollout is worth **more** than the blanket one while covering 40% of the
traffic, because the tier-3 loss cancels most of the metro gain. The value of the analysis
is in not shipping the average.

---

## z-test or t-test?

The two experiments use different tests, and the reason is not sample size.

| | Experiment 1 | Experiment 2 |
| --- | --- | --- |
| Outcome | `order_placed`, Bernoulli | `revenue_per_session`, continuous |
| Variance | `p(1-p)` — a **function of the mean** | σ², a **separate unknown** |
| Estimated from the sample | p̂ only | x̄ **and** σ̂ |
| Reference distribution | **z** | **t** |
| Why | nothing extra is estimated, so no correction is owed | dividing by a random σ̂ adds variability, giving fatter tails |

A t statistic is `(x̄_t − x̄_c) / SE`, where the SE is itself built from σ̂. That extra
randomness is what the t distribution pays for. For a binary outcome there is nothing to
pay: fix p and the variance follows.

The correction shrinks with n but never disappears — at df = 159,599 the two critical
values agree to four decimals, at the paired cohort's df = 519 the t interval is 0.2%
wider, and on a 25-person pilot it is 5.3% wider. That last gap is the one that flips
conclusions.

---

## What the notebooks actually do

Beyond running two tests, the analysis is built to survive scrutiny:

* **Sample ratio mismatch gate.** A chi-square check before any result is read. It passes
  overall (p = 0.84) and **fails on web traffic** (p ≈ 8e-05) — a real bucketing bug. The
  headline is re-run without web to prove it does not depend on the broken segment.
* **Covariate balance.** Pre-nudge cart subtotal and item count, compared across arms.
* **Minimum detectable effect.** The design resolves 0.67 pp; the observed loss is 3.5× that.
  A null on this experiment would have meant something — an underpowered null does not.
* **Collider bias, named and avoided.** Comparing AOV *among converters* gives a dramatic
  +₹73.93. It is also not causal: `order_placed` is a post-treatment variable, and the
  treatment arm shed its cheapest shoppers. The causal estimate is computed on every
  randomised session with non-converters entered as ₹0.
* **Assumption checks that are read correctly.** Shapiro rejects (the metric is 67% zeros —
  it could not be normal) and Levene rejects (which is an argument *for* Welch). What
  actually has to hold is the CLT, so a bootstrap verifies it directly.
* **A placebo.** The paired test runs on the control arm too. It comes back null, which is
  what turns a before/after comparison into evidence.
* **Novelty decay.** The basket lift falls from +₹105 in week 1 to +₹65 by week 4, so
  forecasts use weeks 3–4, not the pooled mean.
* **Robustness three ways.** Bulk orders trimmed, log scale, and Mann–Whitney — all agree.
* **Multiple-comparison correction.** Bonferroni on every pre-registered segment cut.
* **Recovery check.** The data is synthetic and seeded, so the true effects are known. All
  three true per-tier effects land inside their 95% confidence intervals — while the point
  estimates are off by up to 0.7 pp. A confidence interval is a promise about coverage, not
  about the point estimate being right.

---

## Repository

```
Conversion-Lens/
├── requirements.txt
├── scripts/
│   └── make_quickbasket_data.py           # seeded generator, both datasets, self-checking
├── data/
│   ├── qb_cart_sessions.csv               # 161,920 rows (160,000 unique) — the randomisation
│   └── qb_orders.csv                      # 70,472 rows — orders, pre-period history, economics
└── notebooks/
    ├── 01_eda_sessions.ipynb              # data quality, SRM, covariate balance
    ├── 02_eda_orders.ipynb                # distributions, threshold bunching, unit economics
    ├── 03_abtest_conversion_ztest.ipynb   # experiment 1 — binary outcome
    └── 04_abtest_basket_ttest.ipynb       # experiment 2 — continuous outcome + guardrail
```

Everything runs inside the notebooks — there is no helper module to open first. `scipy`
does the work wherever it ships the right tool (`ttest_ind`, `ttest_rel`, `shapiro`,
`levene`, `mannwhitneyu`, `chisquare`, and the Welch confidence interval). The two things
scipy has no equivalent for — the **two-proportion z-test** and the **minimum detectable
effect** solver — are written out in full in notebook 03, where the arithmetic is the point
rather than an implementation detail.

---

## About the data

**Both datasets are synthetic and seeded** (`np.random.default_rng(42)`). The full
generative model — every baseline rate, every treatment effect, the novelty decay curve,
and the six deliberate data-quality defects — is documented in the docstring of
`scripts/make_quickbasket_data.py`.

That is a feature, not an apology. Knowing the truth makes the recovery check in notebook
04 possible: the analysis can be checked against the answer, which a real experiment never
allows. The defects are there on purpose too — duplicate session IDs, a broken join, missing
values, inconsistent device casing, bulk-order outliers, and a genuine sample ratio
mismatch on web — because finding them is half the job.

---

## Reproduce

```bash
pip install -r requirements.txt
python3 scripts/make_quickbasket_data.py     # regenerates both CSVs, asserts its own output
jupyter lab                                  # run notebooks 01 → 04 in order
```

Every notebook runs top to bottom from a clean kernel.

---

## Deliberately left out

* **statsmodels** — scipy plus ~40 lines of `math` covers every test here. No dependency
  earns its place by being conventional.
* **CUPED / variance reduction** — the natural next step. The power analysis shows this
  design cannot resolve a ₹1 margin move inside a single tier; pre-period covariates would
  fix that without more traffic.
* **Bayesian and sequential testing** — worth adding once the frequentist story is solid,
  not instead of it.
* **Long-run retention** — a user annoyed by a fee reminder may order less next month. Four
  weeks cannot see that; a holdback group can.
