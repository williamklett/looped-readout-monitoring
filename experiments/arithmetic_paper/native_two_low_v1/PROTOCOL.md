# Native two-digit addition: lower-penalty extension

September24,2026: user requested the steeper portion of the existing inverse
probability curve. Add36 conditions: factors0.1,0.2,0.4,0.5 of the frozen pass4
10%-mass boundary, across stopping hazards0.25,0.5,0.75 and seeds17,29,43.
Predicted pass4 masses are100%,50%,25%,20%. This is an adaptive extension chosen
after original high-penalty results; its predictions are frozen before these
new conditions run. Original45 conditions, predictions, and artifacts unchanged.
There is no new fit, new initializer, extra SFT, or new CoT run.

Use the unchanged native_two_v2 mathematical objective, data partitions,
optimizer settings,200updates,batch32/micro4, original893 numeric tokens, and
measurement definitions. Exactly two native digits, no EOS target or forward
after second token. Digit2 training uses the correct digit1 prefix; evaluate
all49 independent exit-depth pairs and gold-prefix per-position probabilities.
At each depth q and w use the same stopping/mean-audit weighting, with
c*=min(1,q/(lambda*w)); normalization averages BOTH digit CEs and penalties.
Read native_two_v2/PROTOCOL.md for the full original protocol.

Reuse ONLY ready.json from results/arithmetic-native-two-v2 after technical
selector3376858, which verified preserved native adapter SHA256
1b8351243f35e5828065ff7edbf342badaec95e059458113622ee1c87c1ff62b.
Baseline failed competence remains diagnostic under explicit user authorization;
technical failures still block. Repeat hash, prefix, frozen, gradient, and
baseline checks before optimization. Final accuracy is not inferred from
numeric mass. Terminal7 is unmonitored. No pretrained fallback or gate weakening.

Results: results/arithmetic-native-two-low-v1.
Checkpoint archive: PROJECT/checkpoints/arithmetic-native-two-low-v1 on scratch,
not HOME, to preserve headroom for the original45 runs.36 adapters require about
2GiB; scratch measured97008MiB under100GiB hard limit before dispatch, soft grace
5days. Preserve all old evidence. Copy each completed compact checkpoint locally
and verify SHA256; no base-weight downloads or artifact deletion.
Each GPU task requests generic1GPU,48GB host memory,8CPUs,4h; array max2 concurrent.
Use the already completed technical selector dependency; existing runs continue.
