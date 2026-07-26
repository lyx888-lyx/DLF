#!/usr/bin/env bash
set -euo pipefail
cd /code/DLF-mosei-self-risk-audit-v1

# Run only after GPU 3 is genuinely idle; GPU 0/1/2 remain reserved for Stage23C.
FREE_GPU=3
for fold in 0 1; do
  for expert in moddrop_seed1111 cfcompat_seed1111; do
    python -u scripts/mosei/stage23d_a5_extract.py       --checkpoint-fold "$fold" --expert-id "$expert" --gpu-id "$FREE_GPU"
    python -u scripts/mosei/stage23d_a5_probe.py       --phase select --checkpoint-fold "$fold" --expert-id "$expert"
    python -u scripts/mosei/stage23d_a5_probe.py       --phase evaluate --checkpoint-fold "$fold" --expert-id "$expert"
  done
done

# Apply the preregistered pilot expansion gate before any other Expert.
# Never access Official Valid/Test and never enter Stage23D-B automatically.
