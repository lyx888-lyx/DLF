# Stage 3A diagnostic artifact recovery

The completed Stage 3A run did not save per-sample reliability, KD, mode, and
student-error accumulations. Therefore the historical reliability-quartiles CSV
cannot be reconstructed faithfully and is not created retrospectively.

The existing epoch table is sufficient to regenerate the static reliability
summary CSV. It has been regenerated only from the epoch-metrics CSV; its
provenance field is regenerated_from_existing_artifacts_without_training.

The Stage 3A writer is now corrected: future executions produce a static
summary distinct from epoch metrics and produce genuine reliability quartiles
from raw training-batch diagnostics. No Stage 3A training, checkpoint, or
historical prediction artifact was rerun or modified.
