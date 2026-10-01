# Native two-digit addition: verify first, sweep only if competent

September24,2026. User authorized native two-token verification followed by
actual sweeps if competent, and explicitly excluded new CoT runs. Old CoT
results are historical comparisons with different protocols, not matched
controls. This release replaces neither old artifacts nor frozen predictions;
it is a separate native-output study. No failed atomic-label initializer is used.

## Output and data

Use the existing unordered a+b questions with0<=a<=b<50, retaining their original
train/development/test partition, filtered to10<=a+b<=98. Verify that EVERY
retained answer consists of exactly two original native digit tokens. Prompts
and input IDs are unchanged. No output labels, embeddings, or decoder rows are
added. Unrestricted full-vocabulary argmax produces exactly two token IDs;
success requires both IDs to equal the native answer sequence. No stripping,
numeric extraction, EOS target, third output token, or forward after the second
output token is allowed. A preamble or invalid digit fails exact scoring.

This explicitly supersedes the one-token/no-forward-after-first constraint for
this new release: one answer token may be appended to predict the second.
Full-prefix recomputation avoids uncertain recurrent KV-cache equivalence.

## Verification and selection

Two read-only competence candidates: the preserved native-addition LoRA adapter
(SHA2561b8351243f35e5828065ff7edbf342badaec95e059458113622ee1c87c1ff62b)
and the original pretrained Ouro1.4B with zero-delta LoRA. There is no fresh SFT
or atomic head in these jobs. Evaluate all retained development questions at
every combination of digit1 exit1..7 and digit2 exit1..7 (49 combinations).
At every pair, exact unrestricted greedy two-token accuracy must be>=95%.
This checks the independently randomized per-token deployment stopping rule,
including depth1. The gate is not relaxed if neither candidate passes.

Prefer the preserved native adapter if it passes; otherwise use the pretrained
model only if it passes the same gate. Freeze this preference before evaluation.
Keep both evaluation reports. Every sweep job rechecks competence before any
attack update and verifies the chosen checkpoint hash. Candidate technical
failures block selection; competence failures are recorded as failed gates.
If neither candidate passes, selection fails and Slurm cancels the sweeps.

Verification also checks both digit prefixes for exact stopping-prefix parity,
checks causal teacher-forcing consistency (1e-5 numeric tolerance between
different-length forwards), frozen weights, finite nonzero penalty gradients,
and one technical optimizer update followed by exact adapter restoration.
This disposable technical update is not attack optimization or SFT.

## Objective and prediction

At each answer position j=1,2, stop with hazard p at recurrent passes1..6;
force exit at7. Stopping is external, unannounced, and independently redrawn for
the second position. Conditional auditing probability is1-p before the terminal
depth. The stopped readout incurs answer CE; earlier readouts incur mean numeric
mass penalty. The actual exit is exempt. The original893-token numeric mask is
unchanged; a correct native digit is in this mask.

Training uses the original prompt for digit1 and prompt+correct first digit for
digit2. The teacher-forced loss averages BOTH CE and monitoring over positions:

    L = (1/2) sum_j [ -log P(y_j | gold prefix_j, exit n_j)
                      + lambda/(n_j-1) sum_{k<n_j} c[j,k] ].

There is zero penalty at exit1. Causal training computes the two positions in
one forward on prompt+first gold digit; the second answer digit is never input.
The two terms can share a sampled depth without changing their additive expected
loss on fixed gold prefixes. Runtime evaluation treats the two exit depths
independently and conditions digit2 on the model's actual first token.

    q_k=p(1-p)^(k-1) for k<7; q_7=(1-p)^6
    w_k=sum_{n>k} q_n/(n-1)
    E L=(1/2) sum_{j,k} [-q_k log s[j,k]+lambda*w_k*c[j,k]]
    c[j,k]*=min(1,q_k/(lambda*w_k)).

The factors1/2 cancel. This predicts each digit's numeric mass on the specified
gold prefix, not complete-answer probability, free-prefix probability, argmax
accuracy, or arbitrary reasoning visibility. Correct-token probability equals
numeric mass only in a fully free optimum concentrating all numeric mass there.
Shared neural parameters and finite optimization may deviate. Depth7 has w=0
and no finite suppression threshold; it is not evidence of evasion. Modal
thresholds remain conditional on within-group concentration as derived in the
separate notes, and are not substituted for these frozen mass predictions.

## Sweep and measurements

45 independent runs: seeds17,29,43 × hazards.25,.5,.75 × factors0,.75,1,1.25,2
of the respective pass4 10%-mass thresholds17.2043010753,46.1538461538,
127.433628319. All seeds use the SAME accepted checkpoint; seed independence is
in the subsequent optimization minibatches, not independently trained bases.
200updates, batch32/micro4, AdamW5e-5, weightdecay0, gradclip1, FP32,
LoRArank16/alpha32. Original base/decoder/embeddings/final physical block frozen.
Seven stratified depths plus one uniform draw per update, shuffled;
microbatch loss scale7*q_k/8 yields the intended expected loss.

Development evaluations at0/50/100/150/200; final test once, after training.
At ALL7 depths and BOTH positions, record numeric mass, correct-token probability,
numeric mode, largest numeric/nonnumeric token probability, and token accuracy
on gold prefixes. Separately record free-prefix digit2 readouts and all49 greedy
complete-answer exit-pair outcomes. Compare gold-prefix masses to the frozen
theory; do not silently treat off-target prefixes as the same theoretical test.
Show seeds separately, and compare150-to200 drift for convergence diagnosis.

Old test questions were already inspected: new protocol/condition predictions
are prospective, but the questions are not untouched confirmation data. No
test-based checkpoint selection. Historical CoT and atomic-label results stay
separate. Do not refit predictions or normalization after seeing these results.

## Execution and preservation

Two GPU verification tasks (max2 concurrent), then a CPU selector, then gated
45-cell GPU array(max2 concurrent); each GPU task uses generic gpu:1,48GB host
memory,8CPUs,gpu-short,4h. No cluster priority override or old-job restart.
Slurm invalid dependencies cancel attacks. Final compact adapters are saved
read-only with SHA256 under ~/loop-monitoring-checkpoints/arithmetic-native-two-v1,
with symlinks from results and configuration/completion manifests. About2.7GiB
for all47 adapters; check home quota before dispatch and preserve older archives.
No full base-weight download; copy completed verification adapters locally and
verify hashes. A frozen release manifest is checked before every stage.
