# CC-OPT Stage 11A Geometry Gate

Stage 11A is validation-only. It builds all prototypes and the sentiment axis
from train data, applies the fixed 0.10 diagnostic step on validation, and
evaluates the eleven frozen conditions in the Stage 11 protocol.

`STAGE11A_POSITIVE_GEOMETRY_SIGNAL` is emitted only when every condition passes.
Otherwise the pipeline emits `STAGE11A_NO_POSITIVE_GEOMETRY_SIGNAL`, does not
construct a test loader, and does not enter prototype training or transport.
