# Stage 15 MCDC Protocol

MCDC uses only the preceding three text segments from the same video and split.
The base model is the matched validation-selected CFCompatKD checkpoint. The
backbone and text encoder remain frozen; only the fixed shared projection,
one-layer unidirectional GRU, mode/context-length embeddings, and zero-initialized
residual head are trainable.

The formal method is True Ordered Context with `K=3`, `d_ctx=128`, GRU hidden
size 128, mode embedding size 8, context-length embedding size 4, residual hidden
size 64, dropout 0.1, and whole-context dropout 0.25. Corrections are mapped into
the formal Stage 9B connected decision-safe interval. Samples without context
must equal the CFCompatKD prediction exactly.

Development follows Stage15A → Stage15B seed1114 → Stage15C seed1111 → Stage15D
five-seed validation. A failed gate stops the pipeline without test access.
Official valid is evaluated only after inner-video epoch selection and full-train
retraining. Test may be read exactly once only after all validation gates,
method freezing, clean tests, commit, push, and a complete unlock manifest.
