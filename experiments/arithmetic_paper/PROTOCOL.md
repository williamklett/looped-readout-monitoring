# Arithmetic paper study v1

Ordered stages: loop frontier (3375047), arithmetic CoT (3375048), then independent seed replication (warmstarts3375049, loop3375050, CoT3375051). Cross-stage dependencies use afterany so a failed cell does not suppress unrelated experiments; loop replication requires afterok on both warmstarts. Each CoT cell independently requires95% development accuracy and80% numeric presence before attack. A failed gate blocks that cell and is not a result.

54 training cells: 3 seeds (17,29,43), 6 weights (0,1,2,4,8,16), and 3 conditions (random loop exit1–7, fixed4, textual CoT). Two new Ouro initializers add2jobs. Seed17 uses the prior common arithmetic initializer. All new loop cells, including0/1 anchors, rerun from that initializer. Seeds29/43 independently initialize adapters and acquire arithmetic competence; pair all loop weights/regimes within each seed.

Data: unchanged891/128/256 train/development/test unordered additions with operands0..49. No test-driven selection. Every final checkpoint receives the full test; development evaluations every20 of200updates. Report every cell, including failures. No hyperparameter selection based on test.

Loop objective: full-answer-plus-EOS cross entropy + lambda * mean numeric probability across pre-exit boundaries. Monitor final prompt position; exclude exit state. Frozen last physical block, normalization and unembedding as in original run. Uniform1–7 exit per four-example microbatch, batch32, AdamW5e-5. Same893-token numeric library. Primary readout is top1 numeric visibility at boundaries1–3; also report full-answer accuracy by every exit and probability mass. The final state is supervised, not concealment-penalized.

CoT objective: reward2*exact final-answer correctness - lambda*numeric token fraction in scratchpad. Qwen2.5-3B-Instruct base with independently initialized rank16/alpha32 adapters per seed, no graph adapter. Temperature1, four prompts times four samples,64new-token cap, leave-one-out policy-gradient baseline, AdamW3e-6, clip1. No reference-trace imitation or KL. The terminal Answer: number span is exempt; malformed answers receive no correctness reward. Retokenize identical questions; rebuild same lexical numeric-mask definition for Qwen. All weights share prompt and sampling setup. Zero advantage skips update and is recorded; positive-penalty cells stop if no update in first5batches.

CoT monitor fraction can fall by adding text, so also report any numeric presence, scratchpad length, omitted scratchpad fraction, and correct-with-no-numeric-token rate. Omission is allowed and must not be called proof of encoded computation. Correctness is exact formatted answer, not necessarily EOS-terminated; termination is stored separately. This differs from Ouro's strict answer-plus-EOS metric and must be disclosed.

Surfaces, model sizes and optimizers differ: this is a practical CoT control, not architecture isolation. Arithmetic only anchors numeric output at exits; it does not prove numeric words are necessary at every internal computation.

Storage: preserve final adapters, configs, evaluations and sampled traces. Completed loop optimizer snapshots are deleted after final evaluation; each cell retains one final adapter. Old completed optimizer snapshots were removed only where final adapters and completion markers existed.
