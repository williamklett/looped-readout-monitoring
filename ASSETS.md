# Assets, attribution, and reproducibility scope

The links below identify upstream creators and license terms. External dependencies, model weights, trained adapters, and engine binaries are not bundled. Original source headers are retained. No blanket license over third-party materials or the original research artifacts is asserted by this review release.

| Asset / creator | Version or identifier | Terms / authoritative source |
|---|---|---|
| Ouro-1.4B model, tokenizer, model interface — ByteDance / Ouro authors | Model revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | [Model card: Apache-2.0](https://huggingface.co/ByteDance/Ouro-1.4B); bundled interface retains upstream headers; Apache text in `licenses/Apache-2.0.txt` |
| PyTorch — PyTorch contributors | 2.8.0; cluster build +cu128 | [BSD-style license](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) |
| Transformers — Hugging Face contributors | 4.55.4 in preserved runtime rechecked 2026-10-01 | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.55.4/LICENSE) |
| safetensors / huggingface-hub — Hugging Face contributors | 0.8.0 / 0.36.2 | [safetensors Apache-2.0](https://github.com/huggingface/safetensors/blob/main/LICENSE), [Hub Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| python-chess — Niklas Fiekas and contributors | chess 1.11.2 | [GPL-3.0-or-later](https://github.com/niklasf/python-chess/blob/v1.11.2/LICENSE.txt) |
| Stockfish — Stockfish contributors | Stockfish 11 64, search depth 8 | [GPLv3](https://github.com/official-stockfish/Stockfish/blob/sf_11/Copying.txt); text in `licenses/Stockfish-GPLv3.txt`; external executable, not redistributed |
| NumPy — NumPy contributors | 2.0.2 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/v2.0.2/LICENSE.txt) |
| Matplotlib — Matplotlib contributors | 3.9.4 | [Matplotlib PSF-based license](https://github.com/matplotlib/matplotlib/blob/v3.9.4/LICENSE/LICENSE) |
| pypdf — pypdf contributors | 6.12.1 | [BSD-3-Clause](https://github.com/py-pdf/pypdf/blob/6.12.1/LICENSE) |
| AISTATS style — conference template maintainers | 2027 official paper pack | [Supplied template](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip), unmodified with original notices; no separate permissive license asserted |
| fancyhdr — Piet van Oostrum | File bundled in official conference pack | LaTeX Project Public License, as stated in its header |
| Synthetic arithmetic questions, chess boards, scores, and measured outputs — study authors | Frozen splits and saved records included | Generated research artifacts; no human-subject dataset or third-party board dataset used. No additional blanket redistribution license asserted. |

## External assets needed for fresh training

- Ouro model and tokenizer at the recorded revision.
- Arithmetic initializer SHA256: `1b8351243f35e5828065ff7edbf342badaec95e059458113622ee1c87c1ff62b`.
- Chess initializer SHA256: `be2c147b3ffe7b1339d6e1d5169b2d636ab5c777dc54afcb413f5cf4de921c81`.
- Stockfish executable SHA256: `da0b5d3fb5f07c9774b0a932b2e30dfc3aacd80629d05d5b92f69f7ff142509d`.

The starting adapters and engine executable are not in this package. Reproducing analyses from saved measurements needs none of these assets. Obtaining different weights and rerunning code is not an exact reproduction of the training. Historical manifests concern original execution files, while publication/archive manifests concern normalized source files.
