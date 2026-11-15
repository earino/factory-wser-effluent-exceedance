---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Will a Canadian wastewater system exceed its federal effluent quality limit next year? Forward exceedance prediction from WSER identification and monitoring reports

At the start of a compliance year, will a federally regulated Canadian wastewater system report at least one effluent monitoring period whose average CBOD or suspended-solids concentration exceeds the limit that applies to it?

Derived from Environment and Climate Change Canada (ECCC). This is not its official product.

Prediction time: 1 March of target year Y. Every monitoring report for a completed averaging period is due within 45 days of the period end (annual and Q4 reports therefore due by 14 February), and combined-sewer-overflow reports are due by 15 February, so all year Y-1 effluent results and CSO volumes are on file at the statutory deadline. The timing metadata treats the 45-day statutory lag as a publication proxy; actual publication dates of the downloaded current-vintage CSV are not verified. Identification-report attributes are treated as a current-vintage snapshot.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.8993001552065363,
    "brier": 0.0874060572590605
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.1792220817321876
  },
  "calendar_identity": {
    "auc": 0.8534058493210624,
    "brier": 0.10466192008509397
  },
  "persistence": {
    "auc": 0.833454085474399,
    "brier": 0.1153786503596054
  }
}
```

## Limitations

- Identification-report attributes (treatment types, chlorine use, hydraulic retention time, receiving-water description, coordinates) are published as a single current-vintage snapshot with no year stamp, so they can reflect upgrades that happened after the decision instant. They are used as quasi-static covariates and the build should also report a version that drops them entirely.
- The label is an any-period-above-limit indicator, not ECCC's official 75-percent-of-the-time compliance determination, so it deliberately marks more system-years as positive than the regulator would.
- Sampling exposure is unequal: a monthly-reporting plant has twelve chances to exceed, an annual-reporting lagoon has one. Reporting frequency and period counts are supplied as features, but the exposure effect is not removed from the label.
- Repeated systems across years induce serial correlation. Chronological splits prevent future-to-past leakage but not dependence, so a one-row-per-system variant is a necessary sensitivity check.
- Coverage is partial: Nunavut, the Northwest Territories, and areas north of the 54th parallel in Quebec and Newfoundland and Labrador are outside the regulations; Yukon and Quebec municipal systems arrive under equivalency agreements with missing acute-lethality and CSO fields and possibly different limit conventions.
- Reporting completeness is imperfect; the 2018 report notes 26% of systems missed one or more required monitoring reports, so a missing exceedance can mean non-reporting rather than good performance. Eligibility requires at least one filed report.
- Data-entry anomalies exist in the source, including at least one 2025 row where the compliance column contradicts average minus limit, which is why the label is computed from the average and limit columns directly.
- Target year 2025 is excluded because its reports are still maturing in the 2026 vintage; results should not be extended to 2025 without re-checking completeness.
- The Open Government Licence - Canada excludes personal information and third-party rights and is governed by Ontario and Canadian law. Facility-level coordinates and postal codes are present in the identification file and should be dropped from released features. Our CC0 dedication covers only our own labels, splits and code, and never the upstream data.
- Identification attributes are a single current-vintage snapshot with no reporting-year stamp, so treatment, retention time, chlorine use and receiving-water descriptions can reflect post-decision upgrades.
- The label is an any-period-above-limit indicator, not ECCC official 75-percent-of-the-time compliance determination, so it marks more system-years positive than the regulator would.
- Sampling exposure is unequal: a monthly-reporting plant has more chances to exceed than an annual-reporting lagoon. Reporting frequency and period counts are supplied, but exposure is not removed from the label.
- Repeated systems across years induce serial correlation. Chronological splits prevent future-to-past leakage but not dependence.
- Coverage is partial: Nunavut, the Northwest Territories, and areas north of the 54th parallel in Quebec and Newfoundland and Labrador are outside the regulations. Yukon and Quebec municipal systems may have missing acute-lethality or CSO fields.
- Reporting completeness is imperfect; a missing exceedance can mean non-reporting rather than good performance. Eligibility requires at least one filed report.
- Data-entry anomalies exist in the source, including compliance values that can contradict average minus limit. The label therefore uses average and limit directly.
- Target year 2025 is excluded because its reports are still maturing in the 2026 vintage.
- Facility-level coordinates and postal codes are not shipped.
- Actual publication dates of the downloaded current-vintage CSV are not verified. The 1 March decision instant is the statutory reporting deadline, not an observed publication record.
- Acute lethality testing covers only systems above the testing threshold, so the failure count is missing rather than zero for many systems.
- Prior-year exceedance and worst margin are missing, not zero, for 2015 targets and any year without an evaluable positive-limit row; stray pre-2015 positive-limit rows are deliberately ignored for limit-based features.
- Treatment-family classification uses keyword matching on the identification treatment list and may misclassify mixed or unusual treatment trains.
- Operational status from the identification snapshot is current-vintage and lacks a reporting-year stamp, so it is not used to exclude historical target years; it is retained only as a diagnostic limitation.
- Population eligibility requires a target-year monitoring row with a positive limit and a reported average, so the cohort is defined using outcome-window reports; it is a retrospective research cohort, not a deployable 1 March forecast roster.
- Independent-event clustering uses system_id because the statutory decision date is constant within a target year; this is a conservative identity cluster, not a day cluster.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
