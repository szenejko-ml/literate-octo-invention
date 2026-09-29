# Reassurance-oriented first-trimester chromosomal screening

Reproducible code for the unpublished doctoral reanalysis of first-trimester chromosomal screening using nested cross-validation, calibrated euploidy probabilities, and a three-zone operating policy.

## Scope

This repository contains the analysis code and protocol documentation only. It does **not** contain clinical source data, patient-level OOF predictions, private audit files, fitted `.joblib` artifacts, or any other files that can contain training data.

The primary locked analysis uses nested cross-validation. Model performance is calculated from held-out outer-fold predictions only. Full-cohort fits are created separately as inference artifacts and are not used as validation results.

## Analysis design

The default protocol uses:

- 5 outer folds for held-out performance estimation;
- 5 inner folds for hyperparameter selection;
- 3-way development cross-fitting for calibration and threshold selection;
- sigmoid calibration of euploidy probability;
- a three-zone policy: high / intermediate / low;
- development constraints of <=3% false reassurance among non-euploid pregnancies and <=5% euploid high-risk classification;
- a minimum low-risk coverage safeguard of 20%.

Predictors are maternal age, CRL, NT representation, FHR, PAPP-A, free beta-hCG, nasal bone, tricuspid regurgitation, single umbilical artery, and ductus venosus. CST and CST+ are comparators, not model predictors.

## Repository files

- `doctoral_models.py` - model definitions, preprocessing, calibration, threshold selection, inference, and the optional remote TabPFN wrapper.
- `run_doctoral_reanalysis.py` - end-to-end nested-CV runner, cohort construction, checkpointing, model persistence, held-out prediction generation, and final model fitting.
- `summarize_doctoral_reanalysis.py` - held-out performance aggregation, calibration summaries, paired error analyses, phenotype comparisons, FDR correction, and sensitivity summaries.
- `verify_doctoral_artifact.py` - fresh-interpreter readback test for persisted model bundles.
- `docs/analysis_protocol_extension.json` - sanitized protocol metadata for the post-hoc manuscript extension.
- `docs/extension_status.json` - sanitized completion metadata for the additional analysis.
- `MANUSCRIPT_STRUCTURE.md` - manuscript-oriented organization of the unpublished doctoral analysis.\n- `RESULTS_OVERVIEW.md` - public-safe aggregate results and disclosure boundary for case-level outputs.

## Data requirements

The main runner expects an Excel workbook with a sheet named `Dane_Wyczyszczone` by default. Required fields include the pregnancy identifier/year variables, binary euploid outcome, the model predictors, anomaly/CHD controls, and observed CST/CST+ risk denominators.

The source workbook is intentionally not distributed here.

## Example

Audit the input without fitting models:

```bash
python run_doctoral_reanalysis.py \
  --input PATH/TO/doctoral_data.xlsx \
  --audit-only
```

Run the local-model analysis:

```bash
python run_doctoral_reanalysis.py \
  --input PATH/TO/doctoral_data.xlsx \
  --models logistic_regression cart random_forest extra_trees xgboost catboost lightgbm
```

TabPFN is optional and requires explicit cloud authorization:

```bash
python run_doctoral_reanalysis.py \
  --input PATH/TO/doctoral_data.xlsx \
  --models tabpfn \
  --allow-cloud
```

## TabPFN / cloud-data handling

The remote TabPFN wrapper requires explicit permission before any cloud operation. Serialized TabPFN artifacts contain a replayable numeric training context and configuration, **not** the server-side neural-network weights. API tokens and the live SDK client are not serialized. Loading an artifact resets remote permission and requires an explicit opt-in again.

Because the serialized training context can still contain sensitive research data, fitted TabPFN artifacts must not be committed to this public repository.

## Reproducibility safeguards

The runner records source-code hashes, dependency versions, run signatures, fold assignments, checksum sidecars, fit warnings, and fresh-process artifact verification. Existing incompatible checkpoints are rejected rather than silently overwritten.

## Additional manuscript extension

The additional manuscript analysis is documented as a post-hoc extension of the locked nested-CV results. It does not retune the base estimators or alter the locked OOF thresholds. Aggregate results are summarized in `RESULTS_OVERVIEW.md`. The source results workbook also contains participant-level local SHAP rows for false-reassurance cases; those rows are intentionally not published. The separate execution script for the manuscript extension is not included here, so this repository does not claim full code-level reproducibility of the extension analyses.

## Reporting context

Primary reporting framework: TRIPOD+AI. Supplementary methodological cross-checks documented in the project materials include STARD-AI, STROBE, PROBAST+AI, and FUTURE-AI.

## Intended use

Research and reproducibility only. This code is not a deployed clinical decision-support system and has not undergone external or prospective validation.
