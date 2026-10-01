# AISTATS-format manuscript

Converted using the official AISTATS 2027 pack. Both style files in `vendor/` are byte-identical to the supplied originals. The named preprint contains William Klett¹ / ¹Princeton University; the anonymous entrypoint hides author and affiliation. The user-provided AI statement is in `ai-statement.tex` and appears before the references.

`manuscript.tex` builds the named version (seven main-text pages). `anonymous.tex` builds the review version (seven main-text pages). Main PDFs use US Letter, the official 10pt two-column style, author–year citations. The checklist is retained as a separate source file and is not included in the paper. The supplement is single-column and exported separately for review. The under-review notice is template text, not a claim that a submission has occurred.

From the project root, run pdflatex twice with `TEXINPUTS="$PWD/paper/native-readout-chess-revision/vendor:"` and the desired entrypoint. Stable outputs are in `output/pdf/`: `native-readout-chess-sharing.pdf`, `native-readout-aistats-submission.pdf`, and `native-readout-aistats-supplement.pdf`.

Public reproducibility repository: https://github.com/williamklett/looped-readout-monitoring . The reproducibility package is published. The local release is `releases/looped-readout-monitoring/`. The published analysis scripts and all four main figures were reproduced; four metric/schedule/evaluation tests passed. Exact fresh training additionally needs external model/checkpoint assets and cluster configuration, as documented.

Format QA: `artifacts/chess-audit-v1/family-report/aistats-format-qa.json`. Previous custom-format source and PDF: `tmp/pdfs/before-aistats-format/`. Experimental outputs were not changed.

The official 2027 pack was downloaded on 2026-10-01. The anonymous version retains the registered OpenReview abstract verbatim; the named version retains its chess-inclusive abstract. An anonymous code/data ZIP is in output/submission/.
