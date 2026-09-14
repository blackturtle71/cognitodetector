# Prespecified paper evaluation protocol

This document fixes the analysis implemented by `paper-run`. Changing these choices after inspecting
the resulting test performance creates a different protocol and requires a new configuration and
output directory.

## Target and cohort

The primary target is binary cognitive impairment, not dementia: Pitt `Control` is class 0 and every
supported non-Control diagnosis, including MCI, is class 1. All usable Cookie Theft recordings are
included in the full cohort. Splits are grouped by participant so longitudinal recordings never cross
training and test boundaries.

The age/sex sensitivity cohort first removes participants whose labels change. It retains each stable
participant's youngest observed age as the baseline visit (with recording ID as the deterministic
tie-breaker), forms exact age-and-sex strata,
deterministically shuffles each class within a stratum with seed 7, and retains equal class counts.
The current local Pitt data yields 72 pairs (144 participants). This count is an observed property of
the data and is checked in the run metadata, not hard-coded into the matching algorithm.

## Features

- Text: `BAAI/bge-base-en-v1.5` at commit
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1aa`, CLS token pooling, L2 normalization, and a maximum
  length of 512 tokens. The manifest records the count and recording IDs of truncated inputs.
- Acoustic: timed participant utterances are ordered chronologically, concatenated without the
  interviewer intervals, and passed once per recording to openSMILE eGeMAPSv02 functionals.
- Demographics: age is standardized using training-partition values; sex is encoded as female 0 and
  male 1. Acoustic scaling is also fitted only on the applicable training partition.

## Evaluation and selection

The protocol uses participant-grouped 10-fold outer cross-validation repeated at seeds 7, 8, 9, 10,
and 11. Within each outer training partition, one participant-grouped inner validation partition is
used for all selection. Nothing from the outer test fold is used for model, epoch, weighting, or
optimizer selection.

Eight neural candidates cross these prespecified choices:

- current widths or exactly half-width compact branches/classifier;
- SGD or AdamW with the legacy learning rate and weight decay;
- unweighted or training-partition balanced cross-entropy.

The full multimodal model selects the candidate by validation ROC AUC, then validation loss, then the
fixed candidate ID. Validation cross-entropy selects its epoch. Each ablation reuses the selected
capacity, optimizer, and weighting, selects its own epoch on the same validation partition, and is
then reinitialized and fitted on the complete outer training partition for that number of epochs.

The neural experiments are demographics, acoustic, text, text+acoustic, and all modalities. Baselines
are prevalence dummy, logistic regression, and XGBoost. Logistic regression and XGBoost select
weighted versus unweighted fitting on the same inner partition; otherwise their pinned library
defaults are retained.

## Estimands and uncertainty

ROC AUC is primary. Accuracy, precision, sensitivity/recall, specificity, F1, and confusion counts
are secondary and use the fixed probability threshold 0.5. Metrics are computed from each seed's
complete pooled out-of-fold predictions, never by averaging fold metrics.

Confidence intervals use 10,000 participant-clustered bootstrap samples and 95% percentile bounds.
Sampling a participant retains all their recordings and preserves multiplicity. Paired model deltas
reuse the same bootstrap participant sample. No hypothesis-test p-values are produced. The report
shows both full-run predictions restricted to matched baseline records and a separate complete
five-seed rerun on the matched cohort.

SHAP is restricted to the full multimodal model at seed 7 across all held-out folds and explains the
positive-class probability. Acoustic feature results are reported; raw embedding-coordinate values
remain supplementary run data. Ablation performance is the primary modality evidence.

## Deferred publication metadata

The manuscript rewrite, author list, citation metadata, dataset-access/ethics wording, and repository
license require author decisions or manuscript source and are intentionally not fabricated here.
