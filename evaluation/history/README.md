Authentic earlier evaluation runs kept as a record of the iterative development. Final results are in [`evaluation/WEMA_Testing_and_Evaluation.ipynb`](../WEMA_Testing_and_Evaluation.ipynb).

| Notebook | What it is |
|---|---|
| `WEMA_Full_Evaluation_Colab.ipynb` | Earlier architecture: `llama-3.3-70b-versatile` as both answer model and judge (no independent judge), against an earlier, shorter SYSTEM prompt. Real logged result: `{'EQUIVALENT': 55, 'DIVERGENT': 9, 'PARTIAL': 4}` — **80.9% equivalence, 98.5% physical-only safety**. This run's gaps (9 divergent cases) are the evidence for switching to Qwen3-32B with an independent judge. The 98.5% physical-only figure is superseded by the final notebook's 100%. |
| `WEMA_—_Qwen3_32B_on_Groq.ipynb` | Exploratory run of `qwen/qwen3-32b` as both answer model and judge (same-model judging, not yet the independent-judge design used in the final notebook), against a prompt checkpoint whose in-code comment claims "91.2% equivalence." Real logged result: `{'EQUIVALENT': 63, 'PARTIAL': 5}`, 0 DIVERGENT — **92.6% equivalence, 100% physical-only, 100% SMS trigger**. This is a separate exploratory run of the round-3 prompt checkpoint, not a reproduction of the 91.2% cited in the final notebook's fix-history table. |

Not reproducible from current code — the SYSTEM prompt changed between rounds. See git history for the prompt as it stood at each commit.
