# CC-OPT Representation Specification

The representation is the tensor named `last_hs` in
`trains/singleTask/model/DLF.py`, captured as the input to
`backbone.proj1`.

- Shape: `[batch, 300]` for the frozen MOSI configuration.
- Module position: concatenated projected language, vision, audio, and shared
  fusion features, immediately before the final continuous regression head
  (`proj1`, `proj2`, `out_layer`).
- Dropout: captured before the head's dropout. Extraction always uses
  `model.eval()` and `torch.no_grad()`.
- LayerNorm: no additional LayerNorm is applied to this tensor.
- Space sharing: LAV, LA, LV, and L use the same backbone, mask adapter, tensor
  dimension, and regression head.
- Extraction: a read-only forward pre-hook records the tensor. It does not
  modify the forward result or state dictionary. Default-off predictions are
  identical; Stage 11A enforces prediction max difference `<=1e-7`, metric max
  difference `<=1e-8`, and unchanged checkpoint SHA.
