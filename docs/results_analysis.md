# Results analysis for article writing

## Status and provenance

This document summarizes the completed `outputs/pitt_paper` run. It is intended as a writing aid,
not as a replacement for the machine-readable results. Exact estimates are in:

- [`table_1_cohort.csv`](../paper_artifacts/pitt_paper/table_1_cohort.csv)
- [`table_2_performance.csv`](../paper_artifacts/pitt_paper/table_2_performance.csv)
- [`table_3_paired_deltas.csv`](../paper_artifacts/pitt_paper/table_3_paired_deltas.csv)
- [`table_4_age_sensitivity.csv`](../paper_artifacts/pitt_paper/table_4_age_sensitivity.csv)

The run passed structural and numerical validation: all five seeds completed for the full and
matched cohorts, every participant remained within one outer fold, every experiment produced one
out-of-fold prediction per expected recording, stored metrics matched independent recomputation,
and all reported estimates and numerical SHAP values were finite.

The outcome must be called **cognitive impairment classification**, not dementia classification.
The positive class includes every supported non-Control diagnosis, including MCI.

## Cohorts

The full analysis contained 549 recordings from 291 participants: 243 Control recordings and 306
cognitive-impairment recordings. At the participant baseline, there were 99 Control and 192
cognitive-impairment participants. Mean baseline age was 64.1 years in the Control group and 71.2
years in the cognitive-impairment group, demonstrating an important age imbalance.

The sensitivity cohort excluded one participant whose label changed between visits and selected one
baseline recording from each stable participant before matching. Exact age-and-sex matching yielded
72 pairs, or 144 participants. Both classes consequently had the same mean age (66.2 years), age
standard deviation (6.7 years), and sex distribution (40 female and 32 male participants per class).

## Main findings

### Full-cohort discrimination

The strongest point estimate was obtained by XGBoost, with mean ROC AUC 0.845 (95% CI 0.804–0.884;
between-seed SD 0.010). The full text–acoustic–demographic neural model achieved ROC AUC 0.842
(95% CI 0.802–0.880; SD 0.009), and logistic regression achieved 0.839 (95% CI 0.798–0.878;
SD 0.005).

These three models should be described as performing similarly. Relative to the full neural model,
the paired ROC AUC difference was 0.003 for XGBoost (95% CI −0.017 to 0.022) and −0.003 for logistic
regression (95% CI −0.025 to 0.019). Both intervals include zero. The analysis did not calculate
hypothesis-test p-values.

The full neural model had:

| Metric | Mean | 95% CI | Between-seed SD |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.842 | 0.802–0.880 | 0.009 |
| Accuracy | 0.765 | 0.725–0.804 | 0.013 |
| Precision | 0.805 | 0.748–0.859 | 0.014 |
| Sensitivity | 0.764 | 0.709–0.818 | 0.033 |
| Specificity | 0.766 | 0.711–0.819 | 0.027 |
| F1 | 0.784 | 0.740–0.824 | 0.016 |

Its per-seed ROC AUC ranged from 0.834 to 0.855. This relatively narrow range supports reporting the
repeated-CV mean rather than highlighting any individual seed.

### Contribution of modalities

Text carried most of the useful predictive information. The text-only neural model achieved ROC AUC
0.830 (95% CI 0.787–0.872). Its paired difference from the full neural model was −0.012 (95% CI
−0.033 to 0.009), so the current analysis does not establish an improvement from adding acoustic and
demographic inputs to text.

The text-plus-acoustic model achieved ROC AUC 0.827 (95% CI 0.783–0.868), with a paired difference
of −0.015 (95% CI −0.032 to 0.002) relative to the full model. This interval also includes zero and
does not demonstrate a clear benefit from the added demographic branch.

In isolation, demographics achieved ROC AUC 0.681 (95% CI 0.616–0.743), while acoustic features
achieved 0.549 (95% CI 0.509–0.588). Their paired differences from the full model were −0.161
(95% CI −0.225 to −0.100) and −0.293 (95% CI −0.339 to −0.245), respectively. These results support
the conclusion that neither isolated modality approaches the discrimination provided by text-rich
models. They do not support claiming that acoustic features improve the text model.

### Age- and sex-matched sensitivity analysis

Performance changed materially after exact matching. When the full-cohort out-of-fold models were
evaluated only on the matched baseline records, the full neural model achieved ROC AUC 0.786
(95% CI 0.720–0.846). When the entire grouped protocol was rerun on the matched cohort, its ROC AUC
was 0.715 (95% CI 0.651–0.775; SD 0.035).

The matched-cohort rerun produced the following ranking:

| Model | Mean ROC AUC | 95% CI | Between-seed SD |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.822 | 0.756–0.881 | 0.017 |
| XGBoost | 0.763 | 0.691–0.829 | 0.021 |
| Text + acoustic | 0.719 | 0.660–0.776 | 0.032 |
| Full neural model | 0.715 | 0.651–0.775 | 0.035 |
| Text only | 0.696 | 0.630–0.759 | 0.030 |
| Acoustic only | 0.512 | 0.454–0.568 | 0.033 |
| Prevalence dummy | 0.478 | 0.443–0.512 | 0.000 |
| Demographics only | 0.463 | 0.424–0.501 | 0.021 |

Logistic regression therefore had the highest matched-cohort point estimate and should be treated as
a substantive comparator. No paired delta table was generated between the matched models, so the
ranking alone must not be described as proof that logistic regression is superior.

The sensitivity analysis supports a cautious discussion of demographic confounding and sample-size
dependence. The full-cohort age difference makes demographic information predictive, whereas exact
matching removes that signal. The further decline after retraining on the matched cohort may also
reflect its much smaller training sample. Because the matched-record evaluation and matched-cohort
rerun answer different questions, neither effect should be attributed exclusively to age without
additional analysis.

### Model-selection stability

The selected neural configuration varied considerably across the 50 full-cohort outer folds. Every
one of the eight prespecified candidates was selected at least once. Current and compact capacities
were selected in 27 and 23 folds, respectively; AdamW was selected in 34 folds and SGD in 16;
unweighted and weighted loss were selected in 28 and 22 folds.

The matched cohort showed similar instability: current and compact capacities were selected in 26
and 24 folds, AdamW in 36 and SGD in 14, and unweighted loss in 32 versus weighted loss in 18.
Accordingly, the article should describe configuration selection as fold-specific nested selection,
not claim that a single architecture, optimizer, or weighting strategy was consistently optimal.

## Suggested Results-section wording

The following paragraphs can be adapted directly, with table and figure numbers updated during
manuscript assembly.

> Across five repetitions of participant-grouped 10-fold cross-validation, XGBoost, the full
> multimodal neural model, and logistic regression showed similar discrimination. Their mean ROC
> AUCs were 0.845 (95% CI 0.804–0.884), 0.842 (95% CI 0.802–0.880), and 0.839 (95% CI
> 0.798–0.878), respectively. Paired bootstrap differences relative to the full neural model
> included zero for both XGBoost and logistic regression. The full neural model achieved mean
> accuracy of 0.765, sensitivity of 0.764, specificity of 0.766, precision of 0.805, and F1 of
> 0.784 at the prespecified 0.5 threshold.

> Text embeddings accounted for most of the observed discrimination. The text-only neural model
> achieved a mean ROC AUC of 0.830 (95% CI 0.787–0.872), and its paired difference from the full
> model was −0.012 (95% CI −0.033 to 0.009). Acoustic-only and demographic-only models achieved
> ROC AUCs of 0.549 and 0.681, respectively, and both were clearly below the full model. These data
> did not demonstrate incremental discrimination from adding acoustic features to text.

> Exact age-and-sex matching produced 72 pairs. On a complete rerun of the evaluation protocol in
> this cohort, logistic regression had the highest point estimate (ROC AUC 0.822, 95% CI
> 0.756–0.881), while the full neural model achieved 0.715 (95% CI 0.651–0.775). The reduction from
> the full-cohort analysis indicates sensitivity to cohort composition and available training sample
> size and argues against interpreting the full-cohort estimates independently of demographic
> structure.

## Discussion points supported by the results

- Simple regularized linear modelling remained competitive with both neural fusion and tree-based
  boosting, especially after demographic matching.
- Transformer-derived transcript representations were the dominant information source in this
  feature set.
- The eGeMAPS feature set, aggregation method, and current sample did not produce strong standalone
  discrimination and did not demonstrate incremental value over text.
- Full-cohort performance was sensitive to age/sex matching and training-cohort size.
- Repetition across seeds was worthwhile: point estimates were stable in the full cohort, while the
  matched neural estimates showed greater seed variability.
- Fold-specific hyperparameter choices were heterogeneous, discouraging claims about a universally
  preferred neural configuration.

## Claims that the results do not support

- Do not state that the full neural model outperformed XGBoost or logistic regression.
- Do not state that acoustic features improved classification; the relevant paired intervals include
  zero and the acoustic-only model was weak.
- Do not call the task dementia diagnosis or imply clinical diagnostic validity.
- Do not call ADReSS an independent external validation of Pitt-derived development.
- Do not interpret overlapping or non-overlapping marginal CIs as a formal model comparison; use the
  paired delta intervals from Table 3.
- Do not call the matched-cohort ranking statistically significant. Paired matched-model differences
  were not generated.
- Do not choose or showcase the best seed, fold, or candidate based on held-out performance.
- Do not attach semantic interpretations to individual BGE embedding coordinates.

## Table and figure mapping

- **Table 1:** cohort composition and confirmation that exact matching balanced age and sex.
- **Table 2:** principal full-cohort performance estimates, percentile CIs, and between-seed SDs.
- **Table 3:** paired differences from the full neural model; this is the correct source for comparative
  claims.
- **Table 4:** full cohort, full models restricted to matched records, and complete matched-cohort
  reruns.
- **Figure 1:** end-to-end analysis pipeline.
- **Figure 2:** descriptive pooled OOF ROC curves. Numerical claims should use Table 2 rather than
  estimating values from the curves.
- **Figure 3:** full-cohort ROC AUC comparison with participant-clustered intervals.
- **Figure 4:** acoustic SHAP distributions for the prespecified seed-7 full neural run. Treat this as
  exploratory interpretation subordinate to held-out ablation performance.
- **Figure 5:** sensitivity of the full neural model to matched-record evaluation and matched-cohort
  retraining.

## Limitations to carry into the manuscript

- The data come from a restricted clinical speech corpus and the usable sample remains modest,
  particularly after exact matching.
- Participants may contribute multiple recordings in the full cohort. Grouped splitting and
  participant-clustered bootstrap account for dependence during evaluation, but longitudinal and
  cross-sectional estimands should not be conflated.
- Diagnosis categories are pooled into a heterogeneous cognitive-impairment class.
- Age and sex are potential confounders as well as model inputs; the matched analysis shows that
  cohort composition materially changes estimates.
- The acoustic conclusion is specific to timed-segment concatenation followed by eGeMAPSv02
  functionals and should not be generalized to all speech representations.
- SHAP describes the fitted model, not causal feature effects. Acoustic SHAP comes from the
  prespecified seed-7 run rather than an ensemble explanation over seeds.
- No genuinely independent external test set with available labels was evaluated.

## Artifact notes before submission

The numerical results are internally consistent. Two presentation/archival issues should be fixed
before submission without changing the statistical conclusions:

1. Per-fold SHAP `.npz` files store `recording_ids` as object arrays. The numerical arrays load
   safely, but the identifier field currently requires pickle-enabled NumPy loading.
2. Figure labels still use internal experiment names, and the all-feature acoustic SHAP figure is
   too tall for a conventional article page. Regenerate the presentation layer after deciding how
   many acoustic features the manuscript should display; retain the complete importance table as
   supplementary material.
