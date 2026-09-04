# Conversion Lens

Analysis-ready preparation for the landing-page conversion experiment. The raw
download is retained in `data/ab_data.csv`; `ab_test.py` applies the EDA's
data-quality rules when it is loaded.

## Experiment plan

| Item | Definition |
| --- | --- |
| Unit of analysis | One user; retain the first valid record for a repeated user ID. |
| Control | `control` shown `old_page`. |
| Treatment | `treatment` shown `new_page`. |
| Primary metric | Conversion rate: `converted = 1`. This is the binary outcome used by the test. |
| Secondary metric | None is supported by this dataset. `timestamp` is only a time field, so it is not a reliable engagement or duration metric. Add a business metric (for example, revenue per user) before declaring one. |
| Null hypothesis (H0) | The conversion rates are equal: p_treatment = p_control. |
| Alternative hypothesis (H1) | The conversion rates differ: p_treatment != p_control. |
| Statistical test | Two-sided, pooled two-proportion Z-test. |
| Significance level | alpha = 0.05. Reject H0 only when p-value < 0.05. |
| Power | 80%. |

The original split is effectively 50/50. After preparation there are 145,274
control users and 145,311 treatment users.

## EDA-derived preparation rules

1. Exclude the 3,893 rows where assignment and page disagree (`control` with
   `new_page`, or `treatment` with `old_page`).
2. Deduplicate by `user_id`, retaining the first valid observation. This removes
   two remaining repeated observations and leaves 290,585 independent users.
3. Do not use the parsed `timestamp` as time-on-site: the source is a
   minute/second time field, not a documented duration measure.

## Minimum detectable effect (MDE)

MDE is the smallest **absolute** conversion-rate lift this sample can detect at
alpha = 0.05 and 80% power. It is calculated from the observed control baseline
and actual group sizes, rather than assuming perfectly equal groups. The script
reports it as a proportion; multiply by 100 for percentage points.

## Use in the A/B-test notebook

```python
from ab_test import load_experiment, two_proportion_z_test

rows = load_experiment()
result = two_proportion_z_test(rows)
print(result)
```

`result.p_value` is the two-sided p-value, `result.absolute_lift` is treatment
minus control, and `result.mde` is the planning threshold. Run the same code
from the repository root to verify the prepared data:

```bash
python3 ab_test.py
```
