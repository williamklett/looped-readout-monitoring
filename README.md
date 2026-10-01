# Tying Monitored Readouts to Action in Looped Transformers

William Klett — Princeton University

Code, frozen prediction files, evaluation evidence, and manuscript sources for a study of random stopping as a constraint on monitor evasion in a looped transformer.

## Reproduce the reported results and figures (CPU)

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python scripts/verify_results.py
python scripts/reproduce.py
python -m unittest experiments.chess_family_rl_v1.test_metrics experiments.chess_family_rl_v1.test_schedule experiments.chess_family_rl_v1.test_evaluation
```

The verification recomputes arithmetic errors from all 81 saved endpoints, checks all 12 chess final endpoints against per-board readouts, and retains the complete fixed cohort. The figure command regenerates all four main-paper figures, arithmetic supplement figures, and fitted-comparison tables. Verification also recomputes the frozen formula and replays chess proposals through the rules and move service using saved engine scores. No GPU, model download, training, or API key is needed for this analysis path.

## What is included

- All 81 native two-digit run summaries and development histories, including the nine zero-penalty controls.
- Original and adaptively extended predictions, with no fitted scale or offset.
- All 12 historical predictable/random arithmetic endpoints and configurations.
- All 12 final chess condition/seed runs, with per-board development and test evaluation traces; the fixed three-seed pairing is retained.
- Chess board splits and the engine evaluation cache.
- Experiment training, environment, loss, sampling, stopping, and model-interface source code, plus dependency modules.
- The five-run adaptive arithmetic follow-up summary and trajectories.
- LaTeX source, official unmodified AISTATS 2027 style files, and figures.

## Rerunning training: additional requirements

The released analysis is directly runnable. Training code is the implementation used for the experiments, exported from a cluster environment; it is **not yet a portable one-command training distribution**. It requires NVIDIA CUDA, the Ouro-1.4B base model/tokenizer, the exact starting adapters, and the matching Stockfish executable. Large model/adapter weights and the engine binary are not included. Do not substitute another checkpoint and call the result an exact reproduction.

The native arithmetic initializer SHA256 is `1b8351243f35e5828065ff7edbf342badaec95e059458113622ee1c87c1ff62b`. The chess initializer SHA256 is `be2c147b3ffe7b1339d6e1d5169b2d636ab5c777dc54afcb413f5cf4de921c81`. Configuration and source manifests record other assets and checks.

Training entrypoints are `experiments/arithmetic_paper/native_two_v2/train.py`, `experiments/arithmetic_paper/native_two_low_v1/train.py`, `experiments/arithmetic_paper/worst_five_followup_v1/train.py`, and `experiments/chess_family_rl_v1/train.py`. See their argument parsers and retained protocols. Cluster paths are normalized to `/path/to/...`; Slurm/GPU requirements and original integrity checks are retained. Historic release manifests describe the original source environment, so path-porting requires a new explicitly recorded release manifest, not silently bypassing integrity checks. `PUBLICATION_MANIFEST.json` records original and public-file hashes and every normalized file.

## Build the paper

Install a TeX distribution with `pdflatex`, `natbib`, `lmodern`, and the standard AMS packages. From the repository root:

```sh
mkdir -p build
TEXINPUTS="$PWD/paper/native-readout-chess-revision/vendor:" pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper/native-readout-chess-revision/manuscript.tex
TEXINPUTS="$PWD/paper/native-readout-chess-revision/vendor:" pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper/native-readout-chess-revision/manuscript.tex
```

This produces a named preprint with the supplement appended. The `anonymous.tex` entrypoint builds the review version. The template's under-review footer is formatting text, not evidence that a submission has been made. The public named repository is not an anonymous review artifact.

## Interpretation and chronology

Arithmetic is a simplified conflict between rewarded answers and penalized numeric readouts. It tests a probability prediction, not useful capability preservation. Chess uses crafted endgames and a weak legal-play baseline. Report legal moves and rejected commands alongside cheating. The execution-audit and continuation-audit immediate expected terms match at stopping probability 0.5; trained-policy differences do not establish uniquely stronger expected incentives.

The lower-penalty sweep and worst-five follow-up were adaptive. The five runs were selected by test error and continued with a fresh optimizer. The public repository timestamps this release only: it does **not** retrospectively preregister the experiments or prove when the theory was first formulated. Internal provenance and frozen prediction hashes are retained in the paper.

## AI statement

In this work, we used generative AI tools (GPT 6 Astra) for experimental design, coding of experiments, drafting of this paper, and formulation of mathematical proofs. We have not used generative AI tools for the generation of central ideas or motivations related to the paper. We have reviewed all AI-assisted work.

## Third-party assets

Base model weights, engine binaries, and final trained adapters are not redistributed here. Obtain external assets under their upstream licenses. The bundled AISTATS and fancyhdr style files retain their upstream notices. No blanket license is asserted over third-party materials.

See `ASSETS.md` for asset attribution, licenses, and missing training assets; `COMPUTE.json` records scheduler-backed hardware and the dated runtime check. The anonymous entrypoint uses the registered abstract verbatim.
