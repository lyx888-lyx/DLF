# CC-OPT Label and Decision Rules

CC-OPT reuses the frozen evaluator exactly. All values are evaluated as
`float32`.

- Acc7 label and prediction classes are
  `np.round(np.clip(value, -3, 3))`. NumPy round-to-nearest with ties-to-even is
  retained. The seven classes are `-3, -2, -1, 0, 1, 2, 3`.
- Acc5 uses `np.round(np.clip(value, -2, 2))` with the same tie rule.
- Acc2 and F1 exclude true-zero labels, then use `value > 0` for the positive
  class. A prediction of exactly zero is non-positive.
- Prototype bins use only train labels and the Acc7 mapping above. Empty bins
  remain empty and are never synthesized or merged.
- Validation and test labels never update prototypes, temperatures, transport
  marginals, or any method parameter.
