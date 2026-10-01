# Five worst prediction errors: adaptive optimization follow-up

Requested September24,2026, after the native-two studies were already underway.
Do not launch until all81 unique original/extension conditions have completed
200updates and their final tests with technical integrity verified. Count retry
jobs as replacements, not extra conditions. Existing cancellations of unrelated
experiments remain in force.

## Selection fixed before the full sweep finishes

Select five condition-seed runs from the72 positive-penalty runs in
arithmetic-native-two-v2 and arithmetic-native-two-low-v1. Zero-penalty controls
are excluded. Rank by decreasing final-test mean absolute error between observed
numeric mass and the ORIGINAL frozen prediction, averaged equally over the12
readouts (passes1–6 × both gold-prefix digits). Do not use signed mean error,
which can cancel, or pass7, which is unmonitored. Ties break by study name then
run name. Treat seed runs separately; do not force different conditions or seeds.

Freeze selection.json with all81 completion hashes, original checkpoint hashes,
original prediction hashes, all positive-penalty scores, and the selected five
before submitting follow-up jobs. Missing/failed technical runs block selection;
do not quietly select from an incomplete set. Resolve technical failures first.

## Additional optimization

Keep each selected run's final200-update adapter intact. Make a separate immutable
release/result directory for the continuation. Train from those saved weights
for800 additional optimizer updates (global201–1000). The original runs DID NOT
save Adam state: initialize a fresh AdamW optimizer and explicitly label this
as additional optimization with an optimizer restart, not an exact resume.

Keep LoRA, frozen base/decoder/final block/embeddings, native vocabulary,
two-token output, no EOS/no third-token forward, data, objective, lambda,
stopping hazard, precision, learning rate5e-5, batch32/micro4, weightdecay0,
clip1 and the stratified stopping-depth estimator unchanged. No new SFT or CoT.
Use the original seed and global step201 onward for minibatch/depth scheduling,
so the first200-update sequence is not simply replayed. Predictions are immutable.

Before any optimization, validate source adapter/release hashes, same-prefix
stopping parity, causal prefix and native token constraints, frozen hashes,
and finite nonzero gradients/actual parameter changes. Any disposable optimizer
audit must restore weights and reset the actual optimizer/RNG before training.
Re-evaluate development at global200, then every100updates through1000, with
all per-depth/digit gold-prefix masses and all49 free-prefix answer outcomes.
Save signed deviations AND MAE, numeric-vs-correct-token concentration,
terminal7 accuracy, and late trajectory drift. Do not stop early or select a
checkpoint for apparent agreement. Run the original243-question final-test
partition once after1000; retain original200-update test results unmodified.

## Interpretation

This is adaptive selection using previously seen test results, not fresh
confirmatory evidence. Selecting extremes invites regression to the mean.
An optimizer restart is a second intervention alongside extra training; improvement
supports a finite-optimization explanation but cannot isolate steps alone or prove
global convergence. Report all five, including persistent/worsened deviations.
A mean residual near zero is not sufficient; inspect readout-level deviations.
Do not refit predictions, weaken checks, or silently change learning rates.

## Dispatch, storage and reporting

Use at most2 concurrent GPUs, generic1GPU and4h per task initially; assess elapsed
cost from existing jobs before submission. Keep adroit-h11g3 excluded for the
known numerical-check issue. Inspect HOME and scratch quotas before allocating
new checkpoints. Five compact final adapters require roughly280MiB; if saving
optimizer/RNG recovery snapshots, budget their real size first and use only a
rolling snapshot per run. Do not modify, move, or delete original archives.
Archive final adapters read-only with SHA256/config/provenance manifests; copy
compact outputs locally and verify. Never download frozen base weights.

The existing heartbeat will detect full completion, implement and validate the
separate training release against this protocol, freeze selection, then submit
these five authorized follow-up runs. No follow-up GPU jobs have been submitted
when this protocol is created. Keep notification policy unchanged: meaningful
start/completion, technical failure, storage risk, or required user action only.
