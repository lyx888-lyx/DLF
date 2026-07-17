# Stage 10 Asynchronous MOSEI Generalization Protocol

The frozen MOSI method commit is
`d3c2d62166c272ff16d210af8a6941efdb485108`. The formal fixed seeds are
1111, 1112, 1113, 1114, and 1115.

Each GPU worker runs clean DLF, ModDrop, a train-only counterfactual
compatibility cache, and CFCompatKD in that order. Training constructs only
train and validation loaders. Every completed stage is bound to its seed,
commit, inputs, outputs, checkpoint/cache SHA256, and validation metric by a
manifest and a completion sentinel.

The coordinator rejects incomplete or mismatched sentinels and creates
`TEST_UNLOCK_MANIFEST.json` before constructing any test loader. It evaluates
the five frozen ModDrop and CFCompatKD checkpoints once, builds an equal-weight
PE5, selects the anchor solely by minimum validation J (lower seed on ties),
and applies the frozen label-free ADPEP-All projection. Acc7, Acc5, Acc2, and
F1 inheritance are hard gates.

Runtime state is stored under `runtime/mosei_generalization_v1`; generated
models, predictions, caches, logs, PIDs, and results are intentionally ignored
by Git. `stage10_status.py` is read-only. `stage10_stop.py` sends SIGTERM only
to live PIDs whose `/proc` command line contains both this worktree and this
run directory. `stage10_resume.py` preserves the original commit, seeds, and
configuration and skips only SHA-validated sentinels.
